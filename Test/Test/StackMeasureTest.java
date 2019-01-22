package Test;

import myStack.StackMeasure;
import org.junit.Test;

import static org.junit.Assert.*;

public class StackMeasureTest {
    @Test
    public void reverseTest() {
        double[][] temp = {{1,2,3},{4,5,6},{7,8,9}};
        double[] expect = {1,2,3,4,5,6,7,8,9};
        double[] end = StackMeasure.reverse(temp);
        assertArrayEquals(expect,end,0.0);
    }
}