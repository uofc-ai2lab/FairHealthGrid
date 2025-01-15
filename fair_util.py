from dataclasses import dataclass
# Fairness Dataset
from aif360.datasets import BinaryLabelDataset
# Fairness metrics
from aif360.metrics import ClassificationMetric
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

def calculate_fairness_score(EOD: float, AOD: float, SPD: float, DI: float, TI: float) -> dict:
    """
    Calculate a general fairness score based on multiple fairness metrics.
    
    Parameters:
    EOD (float): Equal Opportunity Difference
    AOD (float): Average Odds Difference
    SPD (float): Statistical Parity Difference
    DI (float): Disparate Impact
    TI (float): Theil Index
    
    Returns:
    dict: Dictionary containing individual metric evaluations and overall score
    where 0 indicates perfect fairness and 1 indicates maximum unfairness
    """
    # Define optimal values
    optimal_values = {
        'EOD': 0.0,
        'AOD': 0.0,
        'SPD': 0.0,
        'DI': 1.0,
        'TI': 0.0
    }
    
    # Define acceptable ranges
    ranges = {
        'EOD': (-0.1, 0.1),
        'AOD': (-0.1, 0.1),
        'SPD': (-0.1, 0.1),
        'DI': (0.8, 1.2),
        'TI': (0.0, 0.25)
    }
    
    # Check if each metric is within its acceptable range
    evaluations = {
        'EOD': ranges['EOD'][0] <= EOD <= ranges['EOD'][1],
        'AOD': ranges['AOD'][0] <= AOD <= ranges['AOD'][1],
        'SPD': ranges['SPD'][0] <= SPD <= ranges['SPD'][1],
        'DI': ranges['DI'][0] <= DI <= ranges['DI'][1],
        'TI': ranges['TI'][0] <= TI <= ranges['TI'][1]
    }
    
    # Calculate deviation from optimal values (normalized)
    deviations = {
        'EOD': abs(EOD - optimal_values['EOD']) / 0.1,  # normalized by range size
        'AOD': abs(AOD - optimal_values['AOD']) / 0.1,
        'SPD': abs(SPD - optimal_values['SPD']) / 0.1,
        'DI': abs(DI - optimal_values['DI']) / 0.2,     # normalized by range size
        'TI': abs(TI - optimal_values['TI']) / 0.25
    }
    
    # Calculate raw unfairness score
    raw_score = 0.0
    max_possible_score = 0.0
    
    # Each metric contributes up to 0.2 to the score for being out of range
    # and up to 0.8 based on its deviation
    for metric in evaluations:
        if not evaluations[metric]:
            raw_score += 0.2
        raw_score += deviations[metric] * 0.16  # 0.16 * 5 = 0.8 total possible from deviations
        max_possible_score += 0.2 + 0.16  # Maximum possible contribution from each metric
    
    # Normalize score to 0-1 range
    normalized_score = raw_score / max_possible_score
    final_score = min(1.0, max(0.0, normalized_score))
    
    return {
        'overall_score': round(final_score, 3),
        'metric_evaluations': evaluations,
        'deviations': {k: round(v, 3) for k, v in deviations.items()},
        'is_fair': all(evaluations.values())
    }

@dataclass
class BMnames:
    label_names: str
    protected_att: list
    favorable_label: float = 1.0
    unfavorable_label: float = 0.0


class BMInterface:
    def __init__(self, df_train:pd.DataFrame, df_val:pd.DataFrame, df_test:pd.DataFrame, label: str, protected:list):
        # Assuming train, val, test as default order
        self.data_sets = [df_train, df_val, df_test]
        self.BM_attr = BMnames(label_names=label, protected_att=protected)
        self.transform = False
        self.__generate_sets()
        self.__scale_split()

    def __generate_sets(self):
        self.biLData = []
        for data in self.data_sets:
            self.biLData.append(BinaryLabelDataset(
                df=data,
                label_names=[self.BM_attr.label_names],
                protected_attribute_names = self.BM_attr.protected_att,
                favorable_label = self.BM_attr.favorable_label,
                unfavorable_label = self.BM_attr.unfavorable_label
                )
            )

    def __scale_transform(self):
        scaler = StandardScaler()
        self.biLData[0].features = scaler.fit_transform(self.biLData[0].features)
        self.biLData[1].features = scaler.transform(self.biLData[1].features)
        self.biLData[2].features = scaler.transform(self.biLData[2].features)

    def __scale_split(self):
        scaler = StandardScaler()
        scaler.fit(self.biLData[0].features)
        xy_tmp = []
        for i in self.biLData:
            xy_tmp.append(
                (scaler.transform(i.features),
                 i.labels.ravel())
            )
        self.trainXY, self.valXY, self.testXY = xy_tmp

    # change scaler behavior
    def set_transform(self) -> None:
        self.transform = True
        self.__scale_transform()
    
    def get_protected_att(self) -> list:
        return self.BM_attr.protected_att

    def restore_BLD(self) -> None:
        self.__generate_sets()
        if self.transform:
            self.__scale_transform()
        self.__scale_split()

    def get_train_BLD(self) -> BinaryLabelDataset:
        return self.biLData[0].copy(deepcopy=True)
    
    def get_val_BLD(self) -> BinaryLabelDataset:
        return self.biLData[1].copy(deepcopy=True)
    
    def get_test_BLD(self) -> BinaryLabelDataset:
        return self.biLData[2].copy(deepcopy=True)

    def get_train_xy(self):
        return self.trainXY[0], self.trainXY[1]
    
    def get_test_xy(self):
        return self.testXY[0], self.testXY[1]

    def get_val_xy(self):
        return self.valXY[0], self.valXY[1]
    
    def pre_BM_set(self, new_train_BLD:BinaryLabelDataset) -> None:
        self.biLData[0] = new_train_BLD
        if self.transform:
            self.__scale_transform()
        self.__scale_split()

    def pos_bm_set(self, new_test_BLD:BinaryLabelDataset) -> None:
        self.biLData[-1] = new_test_BLD
        if self.transform:
            self.__scale_transform()

class BMMetrics:
    def __init__(self, bmI:BMInterface, class_array:np.ndarray, pred_val:np.ndarray|BinaryLabelDataset,
                 pred_test:np.ndarray|BinaryLabelDataset, privileged_group:list[dict], unprivileged_group:list[dict]):
        self.bmI = bmI
        # positive class index
        self.pos_idx = np.where(class_array == bmI.get_train_BLD().favorable_label)[0][0]
        # deal with transform mode
        self.in_mode = False
        if isinstance(pred_val, BinaryLabelDataset):
            self.in_mode = True

        self.pred_test = pred_test
        self.pred_val = pred_val
        
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group

        num_thresh = 100
        self.balanced_acc = np.zeros(num_thresh)
        self.class_threshold = np.linspace(0.01, 0.99, num_thresh)

        self.__get_thresould()
        self.__set_pred_BLD()

    def set_new_pred(self, pred_val:np.ndarray, pred_test:np.ndarray):
        self.pred_val = pred_val
        self.pred_test = pred_test

        self.__get_thresould()
        self.__set_pred_BLD()
        
    def set_pos_pred(self, test_BLD_pred:BinaryLabelDataset):

        fav_inds = test_BLD_pred.scores > self.best_class_thresh
        test_BLD_pred.labels[fav_inds] = test_BLD_pred.favorable_label
        test_BLD_pred.labels[~fav_inds] = test_BLD_pred.unfavorable_label

        self.cmetrics = ClassificationMetric(self.bmI.get_test_BLD(), 
                                                test_BLD_pred, 
                                                unprivileged_groups=self.unprivileged_group, 
                                                privileged_groups=self.privileged_group
                                            )

    def get_pred_test(self):
        return self.pred_test
    
    def __set_pred_BLD(self):  
        if not self.in_mode:   
            pred_BLD = self.bmI.get_test_BLD()
            pred_BLD.scores = self.pred_test[:, self.pos_idx].reshape(-1, 1)

        for thresh in self.class_threshold:

            if self.in_mode:
                self.pred_test.labels = (self.pred_test.scores >= thresh).astype(int)
                pred_BLD = self.pred_test
            else:
                fav_idx = pred_BLD.scores > thresh 
                pred_BLD.labels[fav_idx] = pred_BLD.favorable_label
                pred_BLD.labels[~fav_idx] = pred_BLD.unfavorable_label

            self.cmetrics = ClassificationMetric(self.bmI.get_test_BLD(), 
                                                pred_BLD, 
                                                unprivileged_groups=self.unprivileged_group, 
                                                privileged_groups=self.privileged_group
                                                )

            if thresh == self.best_class_thresh:
                break

    
    def __get_thresould(self):
        if not self.in_mode:
            pred_val_BLD = self.bmI.get_val_BLD()
            pred_val_BLD.scores = self.pred_val[:, self.pos_idx].reshape(-1, 1)

        for idx, class_thresh in enumerate(self.class_threshold):
    
            if self.in_mode:
                self.pred_val.labels = (self.pred_val.scores >= class_thresh).astype(int)
                pred_val_BLD = self.pred_val

            else:
                fav_idx = pred_val_BLD.scores > class_thresh 
                pred_val_BLD.labels[fav_idx] = pred_val_BLD.favorable_label
                pred_val_BLD.labels[~fav_idx] = pred_val_BLD.unfavorable_label
            
            # computing metrics based on two BinaryLabelDatasets: a dataset containing groud-truth labels and a dataset containing predictions
            cm = ClassificationMetric(self.bmI.get_val_BLD(),     
                                      pred_val_BLD,  
                                      unprivileged_groups=self.unprivileged_group,
                                      privileged_groups=self.privileged_group
                                    )

            self.balanced_acc[idx] = 0.5 * (cm.true_positive_rate() + cm.true_negative_rate())

        best_idx = np.where(self.balanced_acc == np.max(self.balanced_acc))[0][0]
        self.best_class_thresh = self.class_threshold[best_idx]

    def __get_classification_metrics(self):
        balanced_acc = 0.5 * (self.cmetrics.true_positive_rate() + self.cmetrics.true_negative_rate())
        acc = self.cmetrics.accuracy()
        precision = self.cmetrics.precision()
        recall = self.cmetrics.recall()
        f1 = 2 * ((precision * recall)/(precision + recall))

        return balanced_acc, acc, precision, recall, f1

    def __get_fair_metrics(self):

        eq_opp_diff = self.cmetrics.equal_opportunity_difference()
        avg_odd_diff = self.cmetrics.average_odds_difference()
        spd = self.cmetrics.statistical_parity_difference()
        disparate_impact = self.cmetrics.disparate_impact()
        theil_idx = self.cmetrics.theil_index()

        return eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx

    def get_report(self) -> dict:
        eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx = self.__get_fair_metrics()
        balanced_acc, acc, precision, recall, f1 = self.__get_classification_metrics()

        return {'balanced_acc':balanced_acc, 'acc':acc, 'precision':precision, 'recall':recall, 'f1':f1,
                'eq_opp_diff':eq_opp_diff, 'avg_odd_diff': avg_odd_diff, 'spd':spd,
                'disparate_impact':disparate_impact, 'theil_idx':theil_idx}

    def get_score(self):
        eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx = self.__get_fair_metrics()
        score_dict = calculate_fairness_score(eq_opp_diff, avg_odd_diff, spd, disparate_impact, theil_idx)
        return score_dict

    def get_groups(self):
        return self.privileged_group, self.unprivileged_group