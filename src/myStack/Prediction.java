package myStack;

import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.util.ArrayList;

public class Prediction {
        /*
         * 定义评价指标
         */
        private double HammingLoss1 = 0;
        private double AvgCorrect = 0;
        private double Precision = 0;
        private double Recall = 0;
//        double F_Measure = 0;



        private double AvgCorrect2 = 0;
        private double Precision2 = 0;
        private double Recall2 = 0;
//        double F_Measure2 = 0;
        private double HammingLoss2 = 0;
        private ArrayList<Double> predict_1 = new ArrayList<>();
        private ArrayList<Double> real_1 = new ArrayList<>();

       private void count1(double[] temp,String s) {
           double num_1 = 0;
           for(double elem:temp){
               if (elem==1)
                       num_1++;
           }
           if(s.equals("P")){
               predict_1.add(num_1);
           }
       }
       /**
        * @Description 重载count方法
        * @param temp
        * @Return void
        * @Author cuiwei
        * @Date 2019-01-23 10:26
        */
       private void count1(double[] temp) {
           double num_1 = 0;
           for(double elem:temp){
              if (elem==1)
                num_1++;
             }
           real_1.add(num_1);
       }



    /**
     * @Description 返回预测结果的评价指标
     * @param  s，判断条件
     * @Return double[]
     * @Author cuiwei
     * @Date 2019-01-22 10:51
     */
        public double[] getvalue(String s) {
            double[] measure = new double[5];
            if(s.equals("-A")) {
                measure[0] = AvgCorrect/10;
                measure[1] = Precision/10;
                measure[2] = Recall/10;
                measure[3] = HammingLoss1/10;
                measure[4] = (2*measure[1]*measure[2])/(measure[1]+measure[2]);
            }
            else if(s.equals("-B")) {
                measure[0] = AvgCorrect2/10;
                measure[1] = Precision2/10;
                measure[2] = Recall2/10;
                measure[3] = HammingLoss2/10;
                measure[4] = (2*measure[1]*measure[2])/(measure[1]+measure[2]);
            }
            return measure;
        }

        public ArrayList getnum(String s){
            if(s.equals("p"))
                return predict_1;
            else
                return real_1;
          }
        /**
         * @Description TODO 预测结果
         * @param train
         * @param test
         * @param numofCla
         * @Return void
         * @Author cuiwei
         * @Date 2019-01-22 11:02
         */
        public void Predict(Instances train, Instances test, int numofCla,int iterator) throws Exception {

            predict_1.clear();
            real_1.clear();

            train.setClassIndex(train.numAttributes()-1);
            test.setClassIndex(test.numAttributes()-1);
            AdaBoostM1 adaclassifier = new AdaBoostM1();
            Bagging bagclassifier = new Bagging();
            J48 baseClassifier = new J48();
//            RandomForest baseClassifier = new RandomForest();
            Measure m = new Measure();
            DataTransform df = new DataTransform();

            adaclassifier.setClassifier( baseClassifier );
            bagclassifier.setClassifier(baseClassifier);
            bagclassifier.buildClassifier(train);
            adaclassifier.buildClassifier( train );

            double[] predictions = new double[numofCla];
            double[] Real = new double[numofCla];

            int count = 0;
            m.reset();
            for(int i = 0;i<test.numInstances();i++) {

                predictions[count] = adaclassifier.classifyInstance(test.instance(i));
                Real[count] = test.instance(i).classValue();

                if(count == numofCla-1) {
                    count1(predictions,"P");
                    count1(Real);
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
//                    count1(predictions,"P");
//                    count1(Real);
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
        }
}

