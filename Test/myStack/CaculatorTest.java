package myStack;

import org.junit.Before;
import org.junit.Test;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import java.util.ArrayList;

import static org.junit.Assert.*;

public class CaculatorTest {

    Caculator c = null;
    int[] labelindice = {2,3,4};


    @Before
    public void beforClass(){
        c = new Caculator();
    }

    @Test
    public void getlabelsTest() {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("fork"));
        attributes.add(new Attribute("size"));
        attributes.add(new Attribute("sum"));
        attributes.add(new Attribute("avg"));
        attributes.add(new Attribute("weight"));
        Instances intances = new Instances("repo_popular",attributes,0);
        for(int i=0;i<2;i++){
            Instance tmp = new DenseInstance(attributes.size());
            tmp.setValue(0,1);
            tmp.setValue(1,0);
            tmp.setValue(2,1);
            tmp.setValue(3,0);
            tmp.setValue(4,1);
            intances.add(tmp);
        }
        double[][] re = c.getlabels(labelindice,intances);
        double[] result = new double[6];
        double[] expect = {1,0,1,1,0,1};
        for(int i=0;i<re.length;i++){
            int tmp = re[0].length;
            result[i] = re[i][tmp-1];
        }
        assertArrayEquals(expect,result,0.01);
    }


    @Test
    public void reverseTest() {
        double[][] temp = {{1,2,3},{4,5,6},{7,8,9}};
        double[] expect = {1,2,3,4,5,6,7,8,9};
        double[] end = c.reverse(temp);
        assertArrayEquals(expect,end,0.0);
    }


    @Test
    public void creatnewInstanceTest() throws Exception{
        double[][] a = {{1,2,3,4,5,6,7,8,9,0,1},{0,1,2,3,4,5,6,7,8,9,0}};
        Instances result = c.creatnewInstance(a);
        double r1 = result.instance(0).classValue();
        double r2 = result.instance(1).classValue();
        assertEquals(1,r1,0);
        assertEquals(0,r2,0);
    }

}