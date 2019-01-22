package myStack;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MLkNN;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;


public class StackMeasure {


    public static void generateArffFile(Instances instances, String path) {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances);
        try {
            saver.setFile(new File(path));
            saver.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //矩阵降维
    public static double[] reverse(double temp[][]) {
        int len = 0;
        for (double[] element : temp) {
            len += element.length;
        }

        double[] temp1d = new double[len];
        int index = 0;
        for (double[] array : temp) {
            for (double element : array) {
                temp1d[index++] = element;
            }
        }
        return temp1d;
    }


    public static void main(String[] args) throws Exception {
        // TODO Auto-generated method stub
        String arffFile_data = "/Users/cuiwei/experiment/data/scene.arff";
        String xmlFile_data = "/Users/cuiwei/experiment/data/scene.xml";
        MultiLabelInstances dataset = null;
        myStack.ImportsData id = new myStack.ImportsData();
        dataset = new MultiLabelInstances(arffFile_data, xmlFile_data);
        Instances workingset = dataset.getDataSet();
        Prediction p = new Prediction();


        for (int fold = 0; fold < 10; fold++) {
            Instances train = workingset.trainCV(10, fold);
            Instances test = workingset.testCV(10, fold);

            int trainSize = (int) Math.round(train.numInstances() * 70 / 100); //50%作为训练集
            int testSize = train.numInstances() - trainSize;


            Instances train1 = new Instances(train, 0, trainSize);
            Instances train2 = new Instances(train, trainSize, testSize);

            //获取测试集标签
            MultiLabelInstances mlTest = new MultiLabelInstances(test, dataset.getLabelsMetaData());
            int[] labelIndices = dataset.getLabelIndices();

            double[][] Testlabel = new double[mlTest.getNumInstances()][labelIndices.length];

            for (int i = 0; i < test.numInstances(); i++) {
                for (int j = 0; j < labelIndices.length; j++)
                    Testlabel[i][j] = test.instance(i).value(labelIndices[j]);
            }

            double[] Testlabel1d = reverse(Testlabel);


            double[][] OutTestData = new double[Testlabel1d.length][11];

            int count1 = 0;
            for (double labe : Testlabel1d) {
                OutTestData[count1++][10] = labe;
            }


            //获取训练集标签
            MultiLabelInstances mlTrain2 = new MultiLabelInstances(train2, dataset.getLabelsMetaData());

            double[][] label = new double[mlTrain2.getNumInstances()][labelIndices.length];

            for (int i = 0; i < train2.numInstances(); i++) {
                for (int j = 0; j < labelIndices.length; j++)
                    label[i][j] = train2.instance(i).value(labelIndices[j]);
            }

            double[] Train2label1d = reverse(label);


            double[][] OutTrain2Data = new double[Train2label1d.length][11];

            int count2 = 0;
            for (double labe : Train2label1d) {
                OutTrain2Data[count2++][10] = labe;
            }


            //抽样

            for (int ptime = 0; ptime < 10; ptime++) {
                int times1 = 0;
                int times2 = 0;
                ArrayList<double[]> Ttrain2likeli = new ArrayList<double[]>();
                ArrayList<double[]> Testlikeli = new ArrayList<double[]>();
                Instances newdata = id.getTrainingSet(ptime, train1, 1);//抽样
                MultiLabelInstances mlTrain = new MultiLabelInstances(newdata, dataset.getLabelsMetaData());
                MLkNN mlknn = new MLkNN();
                mlknn.build(mlTrain);
                //根据train2构建第二层数据
                for (int j = 0; j < train2.numInstances(); j++) {
                    Instance tmp = train2.instance(j);
                    MultiLabelOutput mlo = mlknn.makePrediction(tmp);
                    double[] confident1d = mlo.getConfidences();
                    Ttrain2likeli.add(confident1d);
                }
                double[][] train2like2d = Ttrain2likeli.toArray(new double[Ttrain2likeli.size()][14]);
                double[] train2like1d = reverse(train2like2d);
                for (double element : train2like1d) {
                    OutTrain2Data[times1++][ptime] = element;
                }
                //获取test在此分类器下的预测值
                for (int j = 0; j < test.numInstances(); j++) {
                    Instance tmp = test.instance(j);
                    MultiLabelOutput mlo = mlknn.makePrediction(tmp);
                    double[] confident1d = mlo.getConfidences();
                    Testlikeli.add(confident1d);
                }
                double[][] Test2d = (double[][]) Testlikeli.toArray(new double[Testlikeli.size()][14]);
                double[] Test1d = reverse(Test2d);
                for (double element : Test1d) {
                    OutTestData[times2++][ptime] = element;
                }
            }
            //创建新数据集保存结果
            ArrayList<Attribute> att = new ArrayList<Attribute>();
            for (int k = 0; k < 11; k++) {
                if (k < 10)
                    att.add(new Attribute(String.valueOf("Feature" + k)));
                else
                    att.add(new Attribute(String.valueOf("class")));
            }

            Instances workTrain = new Instances("workTrain", att, 0);
            Instances workTest = new Instances("workTest", att, 0);
            Instance temp = new DenseInstance(att.size());
            temp.setDataset(workTrain);
            for (double[] element : OutTrain2Data) {
                int count = 0;
                for (double ele : element) {
                    temp.setValue(count++, ele);
                }
                workTrain.add(temp);
            }

            for (double[] element : OutTestData) {
                int count = 0;
                for (double ele : element) {
                    temp.setValue(count++, ele);
                }
                workTest.add(temp);
            }
            //bagging与Adaboost分类

            NumericToNominal filter = new NumericToNominal();
            filter.setInputFormat(workTrain);
            String options[] = new String[2];
            options[0] = "-R";
            options[1] = "11-11";           //类标签转换为nominal型
            filter.setOptions(options);
            Instances worksetTrain = Filter.useFilter(workTrain, filter);
            filter.setInputFormat(workTest);
            Instances worksetTest = Filter.useFilter(workTest, filter);

            worksetTrain.setClassIndex(workTrain.numAttributes() - 1);
            worksetTest.setClassIndex(workTest.numAttributes() - 1);
            //数据集输出
//			 String trainpath = "D:\\knnsearch\\yeast\\train.arff";
//			 String testpath = "D:\\knnsearch\\yeast\\test.arff";
//			 generateArffFile(worksetTrain,trainpath);
//			 generateArffFile(worksetTest,testpath);

            p.Predict(worksetTrain, worksetTest, 6);//输入类标签个数 手动改


        }

        double[] Adamesaure = p.getvalue("-A");
        double[] Bagmesaure = p.getvalue("-B");
        System.out.println("AdaBoost Accuracy:" + Adamesaure[0] + " Precision:" + Adamesaure[1] + " Recall:" + Adamesaure[2] +
                " HL:" + Adamesaure[3] + " F-measure：" + Adamesaure[4]);
        System.out.println("Bagging Accuracy:" + Bagmesaure[0] + " Precision:" + Bagmesaure[1] + " Recall:" + Bagmesaure[2] +
                " HL:" + Bagmesaure[3] + " F-measure:" + Bagmesaure[4]);
    }

}
