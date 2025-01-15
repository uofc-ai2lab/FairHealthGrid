from fair_util import BMInterface, BMMetrics
from fair_bm import BMManager, BMType
from fair_log import csvLogger
import numpy as np
from datetime import datetime
from dataclasses import dataclass   
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

#TODO adapt model fit and predict probabilites to an abstract interface

def get_model_proba(model, bmI: BMInterface) -> tuple[np.ndarray]:
    x_train, y_train = bmI.get_train_xy() 
    x_val , y_val = bmI.get_val_xy()
    x_test, _ = bmI.get_test_xy()

    if model.__str__().startswith('LogisticRegression'):
        model = model.fit(x_train, y_train, sample_weight=bmI.get_train_BLD().instance_weights)
        y_val_pred = model.predict_proba(x_val)
        y_test_pred = model.predict_proba(x_test)

    elif any([model.__str__().startswith(i) for i in ('XGBClassifier', 'MLP')]):
        model = model.fit(x_train, y_train)
        y_val_pred = model.predict_proba(x_val)
        y_test_pred = model.predict_proba(x_test)

    elif model.__str__().startswith('TabNet'):
        model.fit(x_train, y_train,
                 eval_set=[(x_val, y_val)]
                )
        y_val_pred = model.predict_proba(x_val)
        y_test_pred = model.predict_proba(x_test)

    else:
        model = model.fit(x_train, y_train, eval_set=[(x_val , y_val)])
        y_val_pred = model.predict_proba(x_val)
        y_test_pred = model.predict_proba(x_test)
        
    return (y_val_pred, y_test_pred)


@dataclass
class dummy_model:
    classes_ = np.array([0,1])

class BMGridSearch:
    def __init__(self, bmI:BMInterface, model, bm_list:list[list[BMType]], privileged_group:list[dict], unprivileged_group:list[dict]):
        self.bmI = bmI
        self.bmMR = BMManager(self.bmI, privileged_group, unprivileged_group)
        self.model = model
        if model is None:
            # prepare BMI to deal with transform classifier (scaler)
            self.bmI.set_transform()
            
        self.bm_list = bm_list
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        self.is_model_in = False

    def __in_model_run(self):
        infer_model = self.model.fit(self.bmI.get_train_BLD())
        y_val_pred = self.model.predict(self.bmI.get_val_BLD())
        y_test_pred = self.model.predict(self.bmI.get_test_BLD())

        return infer_model, y_val_pred, y_test_pred

    def __warmup(self):

        if self.is_model_in:
            _, y_val_pred, y_test_pred = self.__in_model_run()
        else:
            y_val_pred, y_test_pred = get_model_proba(self.model, self.bmI)

        self.bmM = BMMetrics(self.bmI, dummy_model.classes_, y_val_pred, y_test_pred, self.privileged_group, self.unprivileged_group)
        

    def __is_valid_in_processing(self, in_set:list[set[BMType]]) -> tuple[bool, BMType]:
        in_type = None
        enum_count = 0
        for current_set in in_set:
            for item in current_set:
                if item.is_in:
                    enum_count += 1
                    in_type = item
        if enum_count == 0:
                return False, in_type
        return True, in_type

    def run_single_sensitive(self):
        # check if in processing is possible

        is_in, in_type = self.__is_valid_in_processing(self.bm_list)
        if is_in and self.model is not None:
            raise ValueError('In processing BM defined. Combination with classifier is invalid.')

        if is_in:
            self.is_model_in = True
            if in_type == BMType.inAdversarial:
                self.model = self.bmMR.in_AD()
            elif in_type == BMType.inMeta:
                self.model = self.bmMR.in_Meta(self.bmI.get_protected_att()[0])

        # create BMMetric object
        self.__warmup()

        logger = csvLogger(f'experiment_({datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")})')
        experiment_dict = {'model':self.model.__str__().split('(')[0], 'BM':'baseline'}
        experiment_dict.update(self.bmM.get_report())
        experiment_dict.update({'fair_score':self.bmM.get_score()['overall_score']})

        exp_data_list = [experiment_dict]


        for c_set in self.bm_list:
            bm_name = ''
            pre_in_set = [c for c in c_set if c.is_pre]
            in_in_set = [c for c in c_set if c.is_in]
            pos_in_set = [c for c in c_set if c.is_pos]

            for c in pre_in_set:
                bm_name += f' {c.name}'
                if c == BMType.preReweighing:
                    self.bmMR.pre_Reweighing()
                if c == BMType.preDisparate:
                    self.bmMR.pre_DR(self.bmI.get_protected_att()[0])
                if c == BMType.preLFR:
                    self.bmMR.pre_LFR()

            # check if in-processing is in bm_list
            if is_in:
                # clear memory on AD
                self.model.sess.close() 
                if any(in_in_set):
                    if in_type == BMType.inAdversarial:
                        self.model = self.bmMR.in_AD(debias=True)
                        bm_name += ' inAD'
                    elif in_type == BMType.inMeta:
                        self.model = self.bmMR.in_Meta(self.bmI.get_protected_att()[0], tau=0.7)
                else:
                    if in_type == BMType.inAdversarial:
                        self.model = self.bmMR.in_AD()
                _, y_val_pred, y_test_pred = self.__in_model_run()

            else:
                y_val_pred, y_test_pred = get_model_proba(self.model, self.bmI)
            
            self.bmM.set_new_pred(y_val_pred, y_test_pred)

            for c in pos_in_set:
                bm_name += f' {c.name}'
                if c == BMType.posCalibrated:
                    self.bmMR.pos_CEO(self.bmI.get_val_BLD(), self.bmI.get_test_BLD())
                elif c == BMType.posEqqOds:
                    self.bmMR.pos_EO(self.bmI.get_val_BLD(), self.bmI.get_test_BLD())
                elif c == BMType.posROC:
                    self.bmMR.pos_ROC(self.bmI.get_val_BLD(), self.bmI.get_test_BLD())


            new_exp_dict = {'model':self.model.__str__().split('(')[0], 'BM':bm_name[1:]}
            new_exp_dict.update(self.bmM.get_report())
            new_exp_dict.update({'fair_score':self.bmM.get_score()['overall_score']})
            exp_data_list.append(new_exp_dict)

            self.bmI.restore_BLD()

        logger(exp_data_list)
        #aggregate_csv_files('./results/', f'./results/experiment_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}.csv')
                
