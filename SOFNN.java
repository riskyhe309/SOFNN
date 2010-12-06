package craig;

// Robin Gane-McCalla 11.14.10
// This is a prototype of a Self Organizing Neural Network as described in
//An on-line algorithm for creating self-organizing fuzzy neural networks
//(Leng, Prasad, McGinnity 2004)
//it implements the five layers described in that paper but at present
//doesn't learn or add or remove neurons.  Most of the information
//in here comes from page 4 of the paper
import java.util.ArrayList; 
import org.apache.log4j.Logger;


public class SOFNN 
{
    static Logger logger = Logger.getLogger( "SOFNN" );
	static double[] initialCenters = new double[qq.INPUT_VECTOR_LENGTH];
	
	double expectedOutput;
	ArrayList<EBFNeuron> EBFNeurons = new ArrayList<EBFNeuron>();
	ArrayList<normalizationNeuron> normalizationNeurons = new ArrayList<normalizationNeuron>();
	ArrayList<weightedNeuron> weightedNeurons = new ArrayList<weightedNeuron>();

	class  membershipFunction
	{
		public double center;
		public double width;
		
		double output( double input )
		{
			return Math.pow( ( input - center ), 2.0 ) / ( 2 * Math.pow( width, 2.0 ) );			
		}
	}

	// EBFNeurons are the first layer after the input
	class EBFNeuron 
	{
		// An EBF neuron will have multiple membershipFunctions which take
		// values from the input to compute the output of the EBF neuron.
		public membershipFunction[] mf = new membershipFunction[qq.INPUT_VECTOR_LENGTH];
		public double[] mfValues = new double[qq.INPUT_VECTOR_LENGTH];

		public EBFNeuron()
		{
			for( int i=0; i < qq.INPUT_VECTOR_LENGTH; i++ )
			{
				membershipFunction newMF = new membershipFunction();
				newMF.center = initialCenters[i];
				newMF.width = qq.INITIAL_WIDTH;
				mf[i] = newMF;
			}
		}
		
		public double output( double input[] )
		{
			double tempOutput = 0;
			double temp;
			double output;


			// The output of an EBF neuron is the negated sum of its membership functions
			// to the eth power.
			// We subtract the center from the input and square that number
			// then divide the result by two times the width squared, just like in the paper.
			for( int i=0; i < qq.INPUT_VECTOR_LENGTH; i++ )
			{
				temp = Math.pow( ( input[i] - mf[i].center ), 2.0 ) / ( 2 * Math.pow( mf[i].width, 2.0 ) );
				mfValues[i] = Math.exp( -temp );
				tempOutput += temp;
			}
			output = Math.exp( -tempOutput );
			return output;
		}
	}
	
	// The normalization layer makes the sum of all the neurons equal to one
	// with each one weighted according to the percentage of the total output it has.
	class normalizationNeuron
	{		
		public normalizationNeuron()
		{
				
		}
		
		double output( double input, double sum )
		{
			double normalized;
			// Here we return the output of a specific neuron divided by the 
			// sum of all the neurons.
			normalized = input/sum;
//			logger.debug( "Normalized output = " + normalized  );
			return normalized;
		}		
	}

	// The weighted layer multiplies all the outputs by a bias value.
	class weightedNeuron
	{
		public double weight=11181.23;
		
		weightedNeuron()
		{
			
		}
		
		double output( double input )
		{
			double weighted = weight*input;
//			logger.debug( "Weighted output = " + weighted );
			return weighted;
		}
	}
	
	double finalOutput( double[] vector )
	{
		double computedOutput = 0;
		double error;
		double minMFValue;
		int[] minMFIndex = new int[EBFNeurons.size()];
		double EBFSum = 0;
		double[] EBFOutputs = new double[EBFNeurons.size()];
		double[] normalizedOutputs = new double[normalizationNeurons.size()];
		double[] weightedOutputs = new double[weightedNeurons.size()];
		double[][] dist = new double[vector.length][EBFNeurons.size()];
		boolean firingStrengthSatisfied = false;
		double[] firedNeurons = new double[EBFNeurons.size()];
		        
		for ( int i=0; i < EBFNeurons.size(); i++ )
		{
			EBFOutputs[i] = EBFNeurons.get( i ).output( vector );
			EBFSum += EBFOutputs[i];
		}
		
		for ( int i=0; i < normalizationNeurons.size(); i++ )
			normalizedOutputs[i] = normalizationNeurons.get( i ).output( EBFOutputs[i], EBFSum );
		
		for ( int i=0; i < weightedNeurons.size(); i++)
			weightedOutputs[i] = weightedNeurons.get( i ).output( normalizedOutputs[i] );
		
		for( int i=0; i < weightedOutputs.length; i++ )
			computedOutput += weightedOutputs[i];

		for( int i=0; i < EBFNeurons.size(); i++ )
		{
			if( EBFOutputs[i] >= qq.MIN_FIRING_STRENGTH )
			{
				firingStrengthSatisfied = true;
				firedNeurons[i] = EBFOutputs[i];
			}
			else
				firedNeurons[i] = -1;
		}

		error = Math.abs( computedOutput - expectedOutput );
		
        logger.debug( "Neuron fired = " + firingStrengthSatisfied );
        if( firingStrengthSatisfied == true )
        {
    		for( int i=0; i < EBFNeurons.size(); i++ )
    		{
    			if( firedNeurons[i] >= 0 )
    		        logger.debug( "   Neuron[" + i + "] value = " + firedNeurons[i] );
    		}    		
        }
        
        logger.debug( "output result = " + computedOutput + " Error = " + error );
        reportNeurons( EBFOutputs, error );
        
		// Criterion A       
		// No adjustment needed
		if( (  error <= qq.ERROR_TOLERANCE ) && ( firingStrengthSatisfied == true ) )
		{
	        logger.debug( "-----------" );
	        logger.debug( "Criterion A" );
		}
		
		// Criterion B
		// Width needs adjustment
		else if( ( error <= qq.ERROR_TOLERANCE ) && ( firingStrengthSatisfied == false ) )
		{
	        logger.debug( "-----------" );
	        logger.debug( "Criterion B" );
        	for( int i=0; i < EBFNeurons.size(); i++  )
        	{
        		for( int j=0; j < EBFNeurons.get( i ).mf.length; j++ )
        		{
            		minMFValue = EBFNeurons.get(i).mfValues[0];
        			for( int k=1; k < vector.length; k++)
        			{        				
	        			if( EBFNeurons.get( i ).mfValues[k] < minMFValue )
	        			{
	        				minMFValue = EBFNeurons.get( i ).mfValues[k];
	        				minMFIndex[i] = k;
	        			}
	        		}
	        	}
	        }
        	
        	for( int i=0; i < EBFNeurons.size(); i++  )
        	{
    	        logger.debug( "Neuron[" + i + "] MF Min Value = " + EBFNeurons.get( i ).mfValues[ minMFIndex[i] ] );	
        	}
		}
		
		// Criterion C
		// New neuron needed
		else if( ( error > qq.ERROR_TOLERANCE ) && ( firingStrengthSatisfied == true ) )
		{
	        logger.debug( "-----------" );
	        logger.debug( "Criterion C" );
	        
	        // Find the distance between the vector values and the center of 
	        // all of the corresponding membership functions in all neurons
        	for( int i=0; i < EBFNeurons.size(); i++  )
        	{
        		for( int j=0; j < vector.length; j++)
	        			dist[j][i] = Math.abs( vector[j] - EBFNeurons.get( i ).mf[j].center );
			}
	        
        	double[] newCenterVector = new double[vector.length];
           	double[] newWidthVector = new double[vector.length];
        	double[] minVal = new double[vector.length];
        	int[] minIndex = new int[vector.length];
        	
        	// Minimize
        	for( int i=0; i < vector.length; i++ )
        	{
        		minIndex[i] = 0;
        		minVal[i] = dist[i][0];
           		for( int j=1; j < EBFNeurons.size(); j++ )
           		{
           			if( Math.min( minVal[i], dist[i][j] ) != minVal[i] )
           			{
           				minVal[i] = dist[i][j];
           				minIndex[i] = j;
           			}           			
           		}        		
        	}
        	
        	// If the distance < distance threshold make the mf center associated with that distance the mf center for the new neuron
        	// Else if distance > distance threshold make the mf center the associated input value 
        	for( int i=0; i < vector.length; i++ )
        	{
        		if( ( EBFNeurons.get( minIndex[i] ).mf[i].center - minVal[i] ) < qq.DISTANCE_THRESHOLD )
        		{
        			newCenterVector[i] = EBFNeurons.get( minIndex[i] ).mf[i].center;
        			newWidthVector[i] = EBFNeurons.get( minIndex[i] ).mf[i].width;
        		}
        		else
        		{
        			newCenterVector[i] = vector[i];
        			newWidthVector[i] = minVal[i];        			
        		}
        	}
        	
	        logger.debug( "New center vector: "  + newCenterVector[0] + ", "  + newCenterVector[1] +  ", "   + newCenterVector[2] );
	        logger.debug( "New width vector: "  + newWidthVector[0] + ", "  + newWidthVector[1] +  ", "   + newWidthVector[2] );
	        
			EBFNeurons.add( new EBFNeuron() );
			for( int i=0; i < vector.length; i++ )
			{
				EBFNeurons.get( EBFNeurons.size() - 1 ).mf[i].center = newCenterVector[i];
				EBFNeurons.get( EBFNeurons.size() - 1 ).mf[i].width = newWidthVector[i];
			}
			normalizationNeurons.add( new normalizationNeuron() );
			weightedNeurons.add( new weightedNeuron() );
		}
		
		// Criterion D
		// Width adjustment needed
		else if( ( error > qq.ERROR_TOLERANCE ) && ( firingStrengthSatisfied == false ) )
		{
	        logger.debug( "-----------" );
	        logger.debug( "Criterion D" );
	        
        	for( int i=0; i < EBFNeurons.size(); i++  )
        	{
        		for( int j=0; j < EBFNeurons.get( i ).mf.length; j++ )
        		{
            		minMFValue = EBFNeurons.get(i).mfValues[0];
        			for( int k=1; k < vector.length; k++)
        			{        				
	        			if( EBFNeurons.get( i ).mfValues[k] < minMFValue )
	        			{
	        				minMFValue = EBFNeurons.get( i ).mfValues[k];
	        				minMFIndex[i] = k;
	        			}
	        		}
	        	}
	        }
        	
        	for( int i=0; i < EBFNeurons.size(); i++  )
    	        logger.debug( "Neuron[" + i + "] MF Min Value = " + EBFNeurons.get( i ).mfValues[ minMFIndex[i] ] );	
		}		

		return computedOutput;
	}
	
	void reportNeurons( double[] EBFOutputs, double error )
	{
		logger.debug( "Number of neurons: " + EBFNeurons.size() );
       	for( int i=0; i < EBFNeurons.size(); i++  )
    	{
			logger.debug( "EBF[" + i + "] output = " + EBFOutputs[i]  );
    		for( int j=0; j < EBFNeurons.get( i ).mf.length; j++ )
        	        logger.debug( "Neuron[" + i + "], MF[" + j +"], MF = " + EBFNeurons.get( i ).mfValues[j] );        				
    	}		
	}
	
	public SOFNN()
	{
		initialCenters[0] = 11036.37;
		initialCenters[1] = 11178.58;
		initialCenters[2] = 11203.55;
		EBFNeurons.add( new EBFNeuron() );
		normalizationNeurons.add( new normalizationNeuron() );
		weightedNeurons.add( new weightedNeuron() );	
	}

	public double compute( double[] vector, double output ) 
	{
		expectedOutput = output;
		double result = finalOutput( vector );
        return result;
			
	}
}

