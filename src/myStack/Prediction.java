package myStack;

import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.util.ArrayList;

public class Prediction {
        /*
         * 定义评价指标
         */

        double HammingLoss1 = 0;
        double AvgCorrect = 0;
        double Precision = 0;
        double Recall = 0;


        ArrayList<boolean[]> pre = new ArrayList<>();
        ArrayList<boolean[]> real = new ArrayList<>();



        double AvgCorrect2 = 0;
        double Precision2 = 0;
        double Recall2 = 0;
        double HammingLoss2 = 0;
        ArrayList<Double> predict_1 = new ArrayList<>();
        ArrayList<Double> real_1 = new ArrayList<>();

       public void count1(double[] temp,String s) {
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
       public void count1(double[] temp) {
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

        public ArrayList getArray(String s){
            if(s.equals("p"))
                return pre;
            else
                return real;
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
        public void Predict(Instances train, Instances test, int numofCla,int fold,double[] mlknnpre) throws Exception {

            predict_1.clear();
            real_1.clear();

//            String path = "/Users/cuiwei/experiment/k=10/fold"+fold+"/Prediction";
//            Caculator c = new Caculator();
//            Instances tep = new Instances(test,0);
//            Attribute mypre1 = new Attribute("adapre");
//            Attribute mypre2 = new Attribute("bagpre");
//            Attribute mlknn = new Attribute("mlknnpre");
//
//
//            tep.insertAttributeAt(mypre1,tep.numAttributes());
//            tep.insertAttributeAt(mypre2,tep.numAttributes());
//            tep.insertAttributeAt(mlknn,tep.numAttributes());

            AdaBoostM1 adaclassifier = new AdaBoostM1();
            Bagging bagclassifier = new Bagging();
            J48 baseClassifier = new J48();
            Measure m = new Measure();
            DataTransform df = new DataTransform();

            adaclassifier.setClassifier(baseClassifier);
            adaclassifier.buildClassifier(train);


            double[] predictions = new double[numofCla];
            double[] Real = new double[numofCla];



            int count = 0;
            m.reset();
            for(int i = 0;i<test.numInstances();i++) {
//                if(adaclassifier.classifyInstance(test.instance(i))!=test.instance(i).classValue())
//                Instance out = new DenseInstance(tep.numAttributes());
//                for(int j=0;j<test.numAttributes();j++){
//                    out.setValue(j,test.instance(i).value(j));
//                }
//                out.setValue(out.numAttributes()-3,adaclassifier.classifyInstance(test.instance(i)));
//                out.setValue(out.numAttributes()-1,mlknnpre[i]);
//
//                tep.add(out);

                predictions[count] = adaclassifier.classifyInstance(test.instance(i));
                Real[count] = test.instance(i).classValue();

                if(count == numofCla-1) {
//                    count1(predictions,"P");
//                    count1(Real);
                    boolean[] predict_label = df.toBool(predictions);
                    boolean[] real_label = df.toBool(Real);
                    pre.add(predict_label);
                    real.add(real_label);
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


            bagclassifier.setClassifier(baseClassifier);
            bagclassifier.buildClassifier(train);
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
//            c.generateArffFile(tep,path);

            AvgCorrect2 += m.getValue("-A");
            Precision2 += m.getValue("-P");
            Recall2 += m.getValue("-R");
            HammingLoss2 += m.getValue("-H");
        }
}

