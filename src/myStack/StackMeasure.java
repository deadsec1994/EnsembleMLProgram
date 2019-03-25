package myStack;

import mulan.classifier.lazy.MLkNN;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;



public class StackMeasure {

    public static void main(String[] args) throws Exception {
        // TODO Auto-generated method stub
        String arffFile_data = "/Users/cuiwei/experiment/data/emotions.arff";
        String xmlFile_data = "/Users/cuiwei/experiment/data/emotions.xml";

        MultiLabelInstances dataset = null;
        ImportsData id = new ImportsData();
        dataset = new MultiLabelInstances(arffFile_data, xmlFile_data);
        Instances workingset = dataset.getDataSet();
        Prediction p = new Prediction();
        Caculator get = new Caculator();
        int numofcla = dataset.getNumLabels();
        int[] labelIndices = dataset.getLabelIndices();


        for (int fold = 0; fold < 10; fold++) {
            Instances train = workingset.trainCV(10, fold);
            Instances test = workingset.testCV(10, fold);
            double[][] OutTestData = get.getlabels(labelIndices, test);
            double[][] OutTrainData = get.getlabels(labelIndices, train);

            for (int ptime = 0; ptime < 10; ptime++) {
                Instances newdata = get.getTrainingSet(ptime, train, 1);  //抽样
                MultiLabelInstances mlTrain = new MultiLabelInstances(newdata, dataset.getLabelsMetaData());
                MLkNN mlknn = new MLkNN();
                mlknn.build(mlTrain);

                OutTrainData = get.Predictionresult(mlknn, numofcla, train, OutTrainData, ptime);
                OutTestData = get.Predictionresult(mlknn, numofcla, test, OutTestData, ptime);
            }
            //创建新数据集保存结果

            Instances worksetTrain = get.creatnewInstance(OutTrainData);
            Instances worksetTest = get.creatnewInstance(OutTestData);


            //数据集输出
//			 String trainpath = "/Users/cuiwei/experiment/knnsearch/scene/c="+fold+"/train.arff";
//			 String testpath = "/Users/cuiwei/experiment/knnsearch/scene/c="+fold+"/test.arff";
//			 generateArffFile(worksetTrain,trainpath);
//			 generateArffFile(worksetTest,testpath);

//            String trainpath = "/Users/cuiwei/experiment/knnsearch/scene/c=" + fold;
//            File file1dir = new File(trainpath);
//            File file1 = new File(file1dir, "AdaCount_1.txt");
//            if (!file1dir.isDirectory())
//                file1dir.mkdir();
//            if (!file1.isFile())
//                file1.createNewFile();
//            FileWriter out1 = new FileWriter(file1, true);


            //类标签平衡
            MyClassBalancer classfilter = new MyClassBalancer();
            Instances balan = classfilter.process(worksetTrain);
//            int c1 = 0, c0 = 0;
//            for (Instance tmp : balan) {
//                if (tmp.classValue() == 1)
//                    c1++;
//                else
//                    c0++;
//            }
//            System.out.println("c1:" + c1 + " ," + "c0:" + c0 + " testnum:" + worksetTest.numInstances());

            p.Predict(balan, worksetTest, numofcla);

//            ArrayList prediction = p.getnum("p");
//            ArrayList real_1 = p.getnum("r");
//            double[] out_p = new double[prediction.size()];
//            double[] out_r = new double[real_1.size()];
//            for (int i = 0; i < prediction.size(); i++) {
//                out_p[i] = (double) prediction.get(i);
//                out_r[i] = (double) real_1.get(i);
//            }
//            for (int j = 0; j < out_p.length; j++) {
//                out1.write(out_p[j] + "," + out_r[j] + "\n");
//            }
//            out1.close();
            System.out.println("fold:" + fold);
        }

        double[] Adamesaure = p.getvalue("-A");
        double[] Bagmesaure = p.getvalue("-B");
        System.out.println("AdaBoost Accuracy:" + Adamesaure[0] + " Precision:" + Adamesaure[1] + " \nRecall:" + Adamesaure[2] +
                " HL:" + Adamesaure[3] + " F-measure：" + Adamesaure[4]);
        System.out.println("\n\nBagging Accuracy:" + Bagmesaure[0] + " Precision:" + Bagmesaure[1] + " \nRecall:" + Bagmesaure[2] +
                " HL:" + Bagmesaure[3] + " F-measure:" + Bagmesaure[4]);
    }

}
