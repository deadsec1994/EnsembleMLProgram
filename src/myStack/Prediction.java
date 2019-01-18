package myStack;

import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class Prediction {
        /*
         * 定义评价指标
         */
        double HammingLoss1 = 0;
        double AvgCorrect = 0;
        double Precision = 0;
        double Recall = 0;
        double F_Measure = 0;



        double AvgCorrect2 = 0;
        double Precision2 = 0;
        double Recall2 = 0;
        double F_Measure2 = 0;
        double HammingLoss2 = 0;



        public double[] getvalue(String s) {
            double[] measure = new double[5];
            if(s.equals("-A")) {
                measure[0] = AvgCorrect/10;
                measure[1] = Precision/10;
                measure[2] = Recall/10;
                measure[3] = HammingLoss1/10;
                measure[4] = F_Measure;
            }
            else if(s.equals("-B")) {
                measure[0] = AvgCorrect2/10;
                measure[1] = Precision2/10;
                measure[2] = Recall2/10;
                measure[3] = HammingLoss2/10;
                measure[4] = F_Measure2;
            }
            return measure;
        }


        public void Predict(Instances train, Instances test, int numofCla) throws Exception {


            train.setClassIndex(train.numAttributes()-1);
            test.setClassIndex(test.numAttributes()-1);

            AdaBoostM1 adaclassifier = new AdaBoostM1();
            Bagging bagclassifier = new Bagging();
            J48 baseClassifier = new J48();

            Measure m = new Measure();
            DataTransform df = new DataTransform();

            adaclassifier.setClassifier( baseClassifier );
            bagclassifier.setClassifier(baseClassifier);
            bagclassifier.buildClassifier(train);
            adaclassifier.buildClassifier( train );

//        System.out.println(test.instance(0).numClasses());
//        System.out.println(train.instance(0).numClasses());
            double[] predictions = new double[numofCla];
            double[] Real = new double[numofCla];

            int count = 0;
            m.reset();
            for(int i = 0;i<test.numInstances();i++) {

                predictions[count] = adaclassifier.classifyInstance(test.instance(i));
                Real[count] = test.instance(i).classValue();
                if(count == numofCla-1) {
                    boolean[] predict_label = df.toBool(predictions);
                    boolean[] real_label = df.toBool(Real);
                    count = 0;
                    m.Accuracy(predict_label, real_label);
                    m.Recall(predict_label, real_label);
                    m.Precision(predict_label, real_label);
                    m.HammingLoss(predict_label, real_label);
                }else
                    count++;
            }

            AvgCorrect += m.getValue("-A");
            Precision += m.getValue("-P");
            Recall += m.getValue("-R");
            HammingLoss1 += m.getValue("-H");
            m.reset();

            for(int i = 0;i<test.numInstances();i++) {

                predictions[count] = bagclassifier.classifyInstance(test.instance(i));
                Real[count] = test.instance(i).classValue();
                if(count == numofCla-1) {
                    boolean[] predict_label = df.toBool(predictions);
                    boolean[] real_label = df.toBool(Real);
                    count = 0;
                    m.Accuracy(predict_label, real_label);
                    m.Recall(predict_label, real_label);
                    m.Precision(predict_label, real_label);
                    m.HammingLoss(predict_label, real_label);
                }else
                    count++;
            }

            AvgCorrect2 += m.getValue("-A");
            Precision2 += m.getValue("-P");
            Recall2 += m.getValue("-R");
            HammingLoss2 += m.getValue("-H");


            F_Measure += (2*Precision*Recall)/(Precision+Recall);
            F_Measure2 += (2*Precision2*Recall2)/(Precision2+Recall2);
        }
}

