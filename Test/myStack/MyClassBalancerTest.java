package myStack;

import org.junit.Test;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.util.ArrayList;

import static org.junit.Assert.*;

public class MyClassBalancerTest {

    @Test
    public void processTest() throws Exception {
        MyClassBalancer bla = new MyClassBalancer();
        /*
            创建一个测试数据集
            最后一个属性为类标签
            两个正类，一个负类
            同时将类标签变为nominal
         */
        NumericToNominal filter = new NumericToNominal();
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("fork"));
        attributes.add(new Attribute("size"));
        attributes.add(new Attribute("sum"));
        attributes.add(new Attribute("avg"));
        attributes.add(new Attribute("weight"));
        Instances intances = new Instances("repo_popular",attributes,0);
        for(int i=0;i<3;i++){
            Instance tmp = new DenseInstance(attributes.size());
            if(i!=2) {
                tmp.setValue(0, 1);
                tmp.setValue(1, 0);
                tmp.setValue(2, 1);
                tmp.setValue(3, 0);
                tmp.setValue(4, 1);
                intances.add(tmp);
            }else{
                tmp.setValue(0, 1);
                tmp.setValue(1, 0);
                tmp.setValue(2, 1);
                tmp.setValue(3, 0);
                tmp.setValue(4, 0);
                intances.add(tmp);
            }
        }
        filter.setInputFormat(intances);
        String options[] = new String[2];
        options[0] = "-R";
        options[1] = "5-5";
        filter.setOptions(options);
        intances = Filter.useFilter(intances, filter);
        intances.setClassIndex(intances.numAttributes()-1);


        Instances result = bla.process(intances);
        double[] re = new double[4];
        for(int i=0;i<4;i++){
            re[i] = result.instance(i).classValue();
        }
        double[] expect = {1,1,0,0};
        assertArrayEquals(expect,re,0);
    }
}