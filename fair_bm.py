from fair_util import BMInterface, BMMetrics
# Fairness Dataset
from aif360.datasets import BinaryLabelDataset
# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover, LFR
from aif360.algorithms.inprocessing import AdversarialDebiasing, MetaFairClassifier
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing, RejectOptionClassification
from aif360.algorithms import Transformer
# ML/Models
import tensorflow.compat.v1 as tf

from enum import Enum

import string
import random

def get_random_string(N:int = 7):
# using random.choices()
# generating random strings
    return ''.join(random.choices(string.ascii_lowercase +
                             string.ascii_uppercase, k=N))

class BMType(Enum):
    preReweighing = 1
    preDisparate = 2
    preLFR = 3
    inAdversarial = 4
    inMeta = 5
    posCalibrated = 6
    posEqqOds = 7
    posROC = 8

    @property
    def is_pre(self):
        return self in frozenset((BMType.preDisparate, BMType.preReweighing, BMType.preLFR))
    
    @property
    def is_in(self):
        return self in frozenset((BMType.inAdversarial, BMType.inMeta))
    
    @property
    def is_pos(self):
        return self in frozenset((BMType.posCalibrated, BMType.posEqqOds, BMType.posROC))


class BMManager:
    def __init__(self, bmI:BMInterface, privileged_group:list[dict], unprivileged_group:list[dict]):
        self.bmI = bmI
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group

    def pre_Reweighing(self)  -> None:
        rw = Reweighing(unprivileged_groups=self.unprivileged_group,
                        privileged_groups=self.privileged_group)

        rw.fit(self.bmI.get_train_BLD())
        train_BLD_rw = rw.transform(self.bmI.get_train_BLD())
        self.bmI.pre_BM_set(train_BLD_rw)

    def pre_DR(self, sensitive_attribute:str) -> None:
        di = DisparateImpactRemover(repair_level=0.1,
                                    sensitive_attribute=sensitive_attribute)

        train_BLD_di = di.fit_transform(self.bmI.get_train_BLD())
        self.bmI.pre_BM_set(train_BLD_di)

    def pre_LFR(self) -> None:
        lfr = LFR(unprivileged_groups=self.unprivileged_group, privileged_groups=self.privileged_group)
        train_BLD_lfr = lfr.fit_transform(self.bmI.get_train_BLD())
        self.bmI.pre_BM_set(train_BLD_lfr)

    def in_AD(self, debias=False) -> AdversarialDebiasing:
        sess = tf.compat.v1.Session()

        ad = AdversarialDebiasing(privileged_groups=self.privileged_group,
                                    unprivileged_groups=self.unprivileged_group,
                                    scope_name=get_random_string(),
                                    debias=debias,
                                    sess=sess,
                                    adversary_loss_weight=1.2,
                                    batch_size=64)
        return ad
    
    def in_Meta(self, sensitive_attribute:str, tau=0):
        meta = MetaFairClassifier(tau=tau, sensitive_attr=sensitive_attribute, type="fdr")
        return meta
    
    def __pos_abstract(self, pos_classifier:Transformer, valid_BLD_pred:BinaryLabelDataset,
                        test_BLD_pred:BinaryLabelDataset) -> BinaryLabelDataset:
        pos_classifier = pos_classifier.fit(self.bmI.get_val_BLD(), valid_BLD_pred)
        test_pos_pred = pos_classifier.predict(test_BLD_pred)
        self.bmI.pos_bm_set(test_pos_pred)
        return test_pos_pred

    def pos_CEO(self, valid_BLD_pred:BinaryLabelDataset, test_BLD_pred:BinaryLabelDataset) -> BinaryLabelDataset: 
        cost_constraint = "fpr"

        CEO = CalibratedEqOddsPostprocessing(unprivileged_groups=self.unprivileged_group,
                                            privileged_groups=self.privileged_group,
                                            cost_constraint=cost_constraint)
        return self.__pos_abstract(CEO, valid_BLD_pred, test_BLD_pred)

    def pos_EO(self, valid_BLD_pred:BinaryLabelDataset, test_BLD_pred:BinaryLabelDataset) -> BinaryLabelDataset:
        EO = EqOddsPostprocessing(unprivileged_groups=self.unprivileged_group, 
                                 privileged_groups=self.privileged_group)
        
        return self.__pos_abstract(EO, valid_BLD_pred, test_BLD_pred)

    def pos_ROC(self,valid_BLD_pred:BinaryLabelDataset, test_BLD_pred:BinaryLabelDataset) -> BinaryLabelDataset:
        ROC = RejectOptionClassification(unprivileged_groups=self.unprivileged_group,
                                         privileged_groups=self.privileged_group)
        
        return self.__pos_abstract(ROC, valid_BLD_pred, test_BLD_pred)