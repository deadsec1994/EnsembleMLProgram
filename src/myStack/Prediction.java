package myStack;

import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
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
//        double F_Measure = 0;


        ArrayList<boolean[]> pre = new ArrayList<>();
        ArrayList<boolean[]> real = new ArrayList<>();



        double AvgCorrect2 = 0;
        double Precision2 = 0;
        double Recall2 = 0;
//        double F_Measure2 = 0;
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
            /*
            train
            0.045341,0.025106,0.024379,0.015278,0.376713,0.174438,0.10968,0.015391,0.005591,0.270415,0
            0.028207,0.136212,0.050046,0.048532,0.040602,0.08031,0.132373,0.021521,0.079634,0.105531,0
            0.944109,0.75497,0.93934,0.940507,0.828324,0.795575,0.908092,0.752403,0.634688,0.536323,1


            (11)0.027589,0.018247,0.008684,0.014685,0.018888,0.026138,0.011102,0.022784,0.021678,0.010472,0
            test
            0.045341,0.025106,0.176511,0.228101,0.342336,0.24772,0.319404,0.20693,0.477911,0.086322,0
            0.230927,0.217516,0.099924,0.301603,0.040602,0.354309,0.132373,0.021521,0.079634,0.105531,0
            0.781521,0.75497,0.610887,0.548043,0.679403,0.695621,0.520809,0.573102,0.430953,0.536323,1

            0.093567,0.20781,0.354212,0.17536,0.350701,0.213711,0.291205,0.518699,0.428502,0.149705,0

            (11)0.027589,0.018247,0.008684,0.014685,0.018888,0.026138,0.011102,0.022784,0.021678,0.010472,0
             */
            predict_1.clear();
            real_1.clear();

            String path = "/Users/cuiwei/experiment/k=10/fold"+fold+"/Prediction";
            Caculator c = new Caculator();
            Instances tep = new Instances(test,0);
            Attribute mypre1 = new Attribute("adapre");
            Attribute mypre2 = new Attribute("bagpre");
            Attribute mlknn = new Attribute("mlknnpre");


            tep.insertAttributeAt(mypre1,tep.numAttributes());
            tep.insertAttributeAt(mypre2,tep.numAttributes());
            tep.insertAttributeAt(mlknn,tep.numAttributes());


//            System.out.println(tep.numAttributes());


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


            double[] predictions = new double[numofCla];
            double[] Real = new double[numofCla];



            int count = 0;
            m.reset();
            for(int i = 0;i<test.numInstances();i++) {
//                if(adaclassifier.classifyInstance(test.instance(i))!=test.instance(i).classValue())
                Instance out = new DenseInstance(tep.numAttributes());
                for(int j=0;j<test.numAttributes();j++){
                    out.setValue(j,test.instance(i).value(j));
                }
                out.setValue(out.numAttributes()-3,adaclassifier.classifyInstance(test.instance(i)));
                out.setValue(out.numAttributes()-1,mlknnpre[i]);

                tep.add(out);

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

            for(int i = 0;i<test.numInstances();i++) {
//                if(bagclassifier.classifyInstance(test.instance(i))!=test.instance(i).classValue())
//                    tep.add(test.instance(i));
                tep.instance(i).setValue(tep.numAttributes()-2,bagclassifier.classifyInstance(test.instance(i)));

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

