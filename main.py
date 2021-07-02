# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 02:45:58 2021

@author: T90
"""


import sys 
sys.path.append('/Method1_textCNN')
sys.path.append('/Method2_textCNN') 

from Method1_textCNN import Method1_textCNN_predict
from Method2_textCNN import Method2_textCNN_predict

class Predictor:
    def __init__(self,model_name):
        self.model_name = model_name
    def predict(self, fact):
        if(self.model_name=='Method1_textCNN'):
            return Method1_textCNN_predict.predict(fact)
        elif(self.model_name=='Method2_textCNN'):
            return Method2_textCNN_predict.predict(fact)
    
    


fact = '原告：胡某某，男，1963年1月20日出生，汉族，住安徽省安庆市开发区。 委托代理人：王静，安徽益上律师事务所律师。 委托代理人：刘和兵，安徽益上律师事务所实习律师。 被告：冯某某，男，1956年9月9日出生，汉族，住安徽省安庆市怀宁县。 被告：安徽独秀纺织有限公司，住所地安徽省安庆市怀宁县工业园月山大道。 法定代表人：冯某某1，经理。 被告：冯某某2，男，1981年3月3日出生，汉族，住安徽省安庆市怀宁县。\n\n 原告胡某某向本院提出诉讼请求：1、判令被告冯某某、安徽独秀纺织有限公司归还借款550万元及利息（利息自2014年6月10日起计算至清偿之日止，按月利率2％计算）；2、判令被告冯某某2对被告冯某某、安徽独秀纺织有限公司的借款及利息承担连带清偿责任；3、由三被告承担本案的全部诉讼费用。事实与理由：被告冯某某、安徽独秀纺织有限公司于2013年12月10日、2014年1月18日两次向原告借款累计730万元，并由被告冯某某2提供连带责任担保，后归还200万元，余款550万元承诺利率为2.4％。余款550万元到期后，被告冯某某、安徽独秀纺织有限公司未履行还款义务，被告冯某某2也未履行担保义务。 被告冯某某、安徽独秀纺织有限公司、冯某某2在法定期限内未提出答辩。 当事人围绕诉讼请求，向本院提供了证据，本院对原告提供的证据予以确认并在卷佐证。对本案事实，本院认定如下： 被告冯某某、安徽独秀纺织有限公司于2013年12月10日、2014年6月6日两次向原告借款累计730万元。2015年7月30日，两被告出具还款承诺书，承诺书载明，两被告尚欠原告借款550万元，于2015年8月2日前还清，利息自2014年6月10日开始按月利率2.4％计算，如未按时归还，每天按逾期金额的千分之五支付违约金。被告冯某某2在承诺书上签字担保，约定担保方式为连带责任担保，担保期限为二年。 \n'
p1 = Predictor('Method1_textCNN')
print('p1预测的借款人基本属性:',p1.predict(fact))

p2 = Predictor('Method2_textCNN')
print('p2预测的出借人基本属性:',p2.predict(fact)[0])
print('p2预测的借款人基本属性:',p2.predict(fact)[1])