package myStack;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class MeasureTest {
    Measure m = null;
    boolean[] bipartition = {true,true,false,false};
    boolean[] truth = {true,false,false,false};
    @Before
    public void beforclassTest(){
        m = new Measure();
    }

    @Test
    public void accuracyTest() {
        m.reset();
        for(int i = 0;i<2;i++) {
            m.Accuracy(bipartition, truth);
        }
        double reslut = m.getValue("-A");
        assertEquals(0.5,reslut,0);
    }

    @Test
    public void precisionTest() {
        m.reset();
        for(int i = 0;i<2;i++){
             m.Precision(bipartition,truth);
        }
        double reslut = m.getValue("-P");
        assertEquals(0.5,reslut,0);
    }


    @Test
    public void recallTest() {
        m.reset();
        for(int i = 0;i<2;i++) {
            m.Recall(bipartition, truth);
        }
        double reslut = m.getValue("-R");
        assertEquals(1,reslut,0);
    }

    @Test
    public void hammingLossTest() {
        m.reset();
        for(int i = 0;i<2;i++){
            m.HammingLoss(bipartition,truth);
        }
        double reslut = m.getValue("-H");
        assertEquals(0.25,reslut,0);
    }
}