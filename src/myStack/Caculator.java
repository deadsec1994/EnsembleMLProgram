package myStack;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class Caculator {
    /**
     * @Description TODO 对多标签数据集获取标签并变为一列
     * @param labelIndices 标签位置索引
     * @param target  需要获取标签的数据集
     * @Return double[][] 第二层数据矩阵，类标签放在最后一列
     * @Author cuiwei
     * @Date 2019-03-19 16:18
     */
    protected double[][] getlabels(int[] labelIndices, Instances target) {
        double[][] label = new double[target.numInstances()][labelIndices.length];

        for (int i = 0; i < target.numInstances(); i++) {
            for (int j = 0; j < labelIndices.length; j++)
                label[i][j] = target.instance(i).value(labelIndices[j]);
        }

        double[] label1d = reverse(label);


        double[][] OutData = new double[label1d.length][11];

        int count1 = 0;
        for (double labe : label1d) {
            OutData[count1++][10] = labe;
        }
        return OutData;
    }

    /**
     * @param temp
     * @Description 矩阵降维
     * @Return double[]
     * @Author cuiwei
     * @Date 2019-02-22 16:46
     */
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


    /**
     * @Description TODO
     * @param learner 分类器
     * @param numcla 类标签个数
     * @param train 数据集
     * @param output 第二层数据，其中最后一列已经插入了类标签
     * @param time   当前循环轮数
     * @Return double[][] 产生好的第二层数据
     * @Author cuiwei
     * @Date 2019-03-20 16:57
     */
    public double[][] Predictionresult (MultiLabelLearner learner, int numcla, Instances train, double[][] output, int time) throws Exception{
        int count = 0;
        ArrayList<double[]> chance = new ArrayList<>();
        for (int j = 0; j < train.numInstances(); j++) {
            Instance tmp = train.instance(j);
            MultiLabelOutput mlo = learner.makePrediction(tmp);
            double[] confident1d = mlo.getConfidences();
            chance.add(confident1d);
        }
        double[][] train2like2d = chance.toArray(new double[chance.size()][numcla]);
        double[] train2like1d = reverse(train2like2d);
        for (double element : train2like1d) {
            output[count++][time] = element;
        }
        return output;
    }


    /**
     * @Description TODO
     * @param conv  第二层数据矩阵
     * @Return weka.core.Instances 产生好的第二层数据集
     * @Author cuiwei
     * @Date 2019-03-20 16:59
     */
    public Instances creatnewInstance(double[][] conv) throws Exception{
        ArrayList<Attribute> att = new ArrayList<>();
        for (int k = 0; k < 11; k++) {
            if (k < 10)
                att.add(new Attribute("Feature" + k));
            else
                att.add(new Attribute("class"));
        }

        Instances data = new Instances("workTrain", att, 0);
        Instance temp = new DenseInstance(att.size());
        temp.setDataset(data);
        for (double[] element : conv) {
            int count = 0;
            for (double ele : element) {
                temp.setValue(count++, ele);
            }
            data.add(temp);
        }
        NumericToNominal filter = new NumericToNominal();
        filter.setInputFormat(data);
        String options[] = new String[2];
        options[0] = "-R";
        options[1] = "11-11"; //将最后一列变为nominal型，'start-end' 列
        filter.setOptions(options);
        data = Filter.useFilter(data, filter);
        data.setClassIndex(data.numAttributes()-1);

        return data;
    }


    /**
     * @Description TODO 随机抽样
     * @param iteration
     * @param m_data 待抽样数据集
     * @param m_Seed  随机种子
     * @Return weka.core.Instances， sub dataset
     * @Author cuiwei
     * @Date 2019-03-20 17:00
     */
    protected  Instances getTrainingSet(int iteration,Instances m_data,int m_Seed ) throws Exception {
//		int bagSize = (int) (m_data.numInstances());
        Instances bagData = null;
        Random r = new Random(m_Seed + iteration);
//		Random r = new Random();
        bagData = m_data.resample(r);

        return bagData;
    }


    /**
     * @param instances 数据集
     * @param path      存储位置
     * @Description 生成.arff文件
     * @Return void
     * @Author cuiwei
     * @Date 2019-02-22 16:53
     */
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
}
