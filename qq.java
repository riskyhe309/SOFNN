// Created by craig
package craig;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.ArrayList;
import craig.SOFNN;

public final class qq 
{
    public static final int INPUT_VECTOR_LENGTH = 3;
    public static final double ERROR_TOLERANCE = 100;
	public static double INITIAL_WIDTH = 100;
    public static final double MIN_FIRING_STRENGTH = 0.1354;
    public static final double WIDTH_ENLARGEMENT_CONSTANT = 1.12;
    public static final double DISTANCE_THRESHOLD = 10;

    public static void main(String[] args) throws IOException
	{
		String strLine;
		String temp[];
		double[] inputVector = new double[INPUT_VECTOR_LENGTH];
		double expectedValue;
		ArrayList<String> closingPrices = new ArrayList<String>();
		

		FileInputStream djia = new FileInputStream( "/Users/craig429/Documents/DJIAin.csv" );
		DataInputStream djiaIn = new DataInputStream( djia );
		BufferedReader djiaFile = new BufferedReader( new InputStreamReader( djiaIn ) );
		strLine = djiaFile.readLine();	// First line is header text, so discard        

		for(;;)
		{
	        strLine = djiaFile.readLine();
	        
	        if( strLine != null )
	        {
			// Remove the date, which contains a comma
			temp = strLine.split( "\"" );
	        
	        // Split out the rest
	        temp = temp[2].split( "," );
	        closingPrices.add( temp[6] );	        
	        }
	        else
	        	break;
		}

		for( int index=0; index < 3; index++)
		{
			for( int i=0; i < INPUT_VECTOR_LENGTH; i++)
				inputVector[i] = Double.valueOf( closingPrices.get( i+index ) );
			expectedValue = Double.valueOf( closingPrices.get( INPUT_VECTOR_LENGTH + index ) );
			SOFNN net = new SOFNN();
			net.compute( inputVector, expectedValue );
		}
	}
}
