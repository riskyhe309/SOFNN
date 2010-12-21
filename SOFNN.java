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
import Jama.Matrix;



public class SOFNN 
{
    static Logger logger = Logger.getLogger( "SOFNN" );
	static double[] initialCenters = new double[qq.INPUT_VECTOR_LENGTH];
	static int trainingIteration;
	static boolean trainingMode;
	
	ArrayList<EBFNeuron> EBFNeurons = new ArrayList<EBFNeuron>();
	ArrayList<normalizationNeuron> normalizationNeurons = new ArrayList<normalizationNeuron>();
	ArrayList<weightedNeuron> weightedNeurons = new ArrayList<weightedNeuron>();
	ArrayList<double[]> trainingPatterns = new ArrayList<double[]>();
	ArrayList<double[]> parameters = new ArrayList<double[]>();
	ArrayList<Double> desiredOutputs = new ArrayList<Double>();
	ArrayList<Double> actualOutputs = new ArrayList<Double>();
	ArrayList<Double> errors = new ArrayList<Double>();
	double[] EBFOutputs = new double[EBFNeurons.size()];
	double[] normalizedOutputs = new double[normalizationNeurons.size()];
	double[] consequentParams = new double[weightedNeurons.size()+1];
	double[] weightedOutputs = new double[weightedNeurons.size()];


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
				mf[i] = newMF;
			}
		}
		
		public double output( double input[] )
		{
			double tempOutput = 0;
			double temp;
			double output;


			// The output of an EBF neuron is the negated product of its membership functions
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
		
		double output( int nodeId )
		{
			double normalized;
			double thisNodeSum= EBFOutputs[nodeId];
			double allNodeSum=0;
			// Here we return the output of a specific neuron divided by the 
			// sum of all the neurons.
			for( int i=0; i < EBFNeurons.size(); i++ )
			{
				allNodeSum += EBFOutputs[i];				
			}
			normalized = thisNodeSum/allNodeSum;
			logger.debug( "Normalized output Neuron[" + nodeId + "] = " + normalized  );
			return normalized;
		}		
	}

	// The weighted layer multiplies all the outputs by a bias value.
	class weightedNeuron
	{		
		weightedNeuron()
		{
			
		}
		
		double output( double[] input, double[] parameters )
		{
			double sum = 0;
			for( int i=0; i < qq.INPUT_VECTOR_LENGTH; i++ )
				sum += ( input[i] * parameters[i+1] );
			sum += parameters[0];
			
			logger.debug( "Weighted output = " + sum );
			return sum;
		}
	}
	
	double finalOutput( double[] vector, double expectedOutput )
	{
		double computedOutput;
		double error;
		double minMFValue;
		int[] minMFIndex = new int[EBFNeurons.size()];
		double[][] dist = new double[vector.length][EBFNeurons.size()];
		boolean firingStrengthSatisfied = false;
		double[] firedNeurons = new double[EBFNeurons.size()];
		        
		EBFOutputs = new double[EBFNeurons.size()];
		normalizedOutputs = new double[normalizationNeurons.size()];
		weightedOutputs = new double[weightedNeurons.size()];
		
		computedOutput = runIt( vector );
		error = Math.abs( computedOutput - expectedOutput );
		
		// Update the arrays
		actualOutputs.add( computedOutput );
		errors.add( error );
		
		// Form the RLS vectors/matrices
		double[] dVals = new double[desiredOutputs.size()];
		for( int i=0; i < desiredOutputs.size(); i++)
			dVals[i] = desiredOutputs.get( i );
		Matrix D = new Matrix( dVals, desiredOutputs.size() );
		logger.debug( "D = " );
		D.print(D.getColumnDimension(), 2 );
		
		double[][] pVals = new double[trainingIteration][qq.INPUT_VECTOR_LENGTH+1];
		for( int i=0; i < trainingIteration; i++)
		{
			pVals[i][0] = 1;
			for( int j=1; j < qq.INPUT_VECTOR_LENGTH+1; j++ )
			{
				pVals[i][j] = trainingPatterns.get( i )[j-1];
			}
		}
		Matrix P = new Matrix( pVals );
		logger.debug( "P = " );
		P.print(P.getColumnDimension(), 2 );
		
		Matrix Q =  ( ( P.transpose() ).times( P ) ).inverse();
		logger.debug( "Q = " );
		Q.print( Q.getColumnDimension(), 2 );
		
		double[] eVals = new double[errors.size()];
		for( int i=0; i < errors.size(); i++)
			eVals[i] = errors.get( i );
		Matrix E = new Matrix( eVals, errors.size() );
		logger.debug( "E = " );
		E.print(E.getColumnDimension(), 2 );

		Matrix theta = Q.times( P ).transpose().times( D );
		logger.debug( "theta = " );
		theta.print( theta.getColumnDimension(), 2 );
		

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

/*        	// Try widen out the widths. Just go around 10 times for now, until we figure out what we're doing.
        	// Ideally, we go until firing strength of at least one neuron is reached.
        	for( int i=0; i < 10; i++ )
        	{
        		// Widen the centers out
        		for( int j=0; j < vector.length; j++ )
        			EBFNeurons.get( i ).mf[ minMFIndex[i] ].center *= qq.WIDTH_ENLARGEMENT_CONSTANT;
        		
        		// Rerun the net, recalc error and firing stats.
        		computedOutput = runIt( vector);
        		error = Math.abs( computedOutput - expectedOutput );
        		firingStrengthSatisfied = false;
        		for( int j=0; j < EBFNeurons.size(); j++ )
        		{
        			if( EBFOutputs[i] >= qq.MIN_FIRING_STRENGTH )
        			{
        				firingStrengthSatisfied = true;
        				firedNeurons[i] = EBFOutputs[i];
        			}
        			else
        				firedNeurons[i] = -1;
        		}
        		
        		// Are we there yet?
        		if( ( error > qq.ERROR_TOLERANCE ) && ( firingStrengthSatisfied == false ) )
        		{
        			// Nope. Recalc and widen again.
                	for( int ii=0; ii < EBFNeurons.size(); ii++  )
                	{
                		for( int jj=0; jj < EBFNeurons.get( ii ).mf.length; jj++ )
                		{
                    		minMFValue = EBFNeurons.get(ii).mfValues[0];
                			for( int kk=1; kk < vector.length; kk++)
                			{        				
        	        			if( EBFNeurons.get( ii ).mfValues[kk] < minMFValue )
        	        			{
        	        				minMFValue = EBFNeurons.get( ii ).mfValues[kk];
        	        				minMFIndex[ii] = kk;
        	        			}
        	        		}
        	        	}
        	        }
               		for( int j=0; j < vector.length; j++ )
            			EBFNeurons.get( i ).mf[ minMFIndex[i] ].center *= qq.WIDTH_ENLARGEMENT_CONSTANT;
        		}
        		else
        			// Yep. Outta here.
        			break;
        	}
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
*/
		}		

		return computedOutput;
	}

	// Find the minimum distance between the observation element and the corresponding 
	// MF center.
	public double[] calculateDistanceVector( double[] observation )
	{
		double[] distanceVector = new double[qq.INPUT_VECTOR_LENGTH];
		double distance;
		
		for( int i=0; i < qq.INPUT_VECTOR_LENGTH; i++ )
		{
			distanceVector[i] = Math.abs( observation[i] - EBFNeurons.get( 0 ).mf[i].center );
			for( int j=1; j < EBFNeurons.size(); j++ )
			{
				distance =  Math.abs( observation[i] - EBFNeurons.get( j ).mf[i].center );
				if( distance < distanceVector[i] )
					distanceVector[i] = distance;
			}			
		}
		return distanceVector;
		
	}
	
	public void updateParameters( double[] observation, double desiredValue, double actualValue)
	{
		double error = Math.abs( desiredValue - actualValue );
		
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
	
	public double runIt( double[] vector )
	{
		double computedOutput=0;
		double[] parameters = new double[qq.INPUT_VECTOR_LENGTH+1];

		for( int i=0; i < qq.INPUT_VECTOR_LENGTH+1; i++)
			parameters[i] = .33;
		
		for ( int i=0; i < EBFNeurons.size(); i++ )
			EBFOutputs[i] = EBFNeurons.get( i ).output( vector );
		
		for ( int i=0; i < normalizationNeurons.size(); i++ )
			normalizedOutputs[i] = normalizationNeurons.get( i ).output( i );
		
		for ( int i=0; i < weightedNeurons.size(); i++)
			weightedOutputs[i] = weightedNeurons.get( i ).output( vector, parameters );
		
		for( int i=0; i < weightedOutputs.length; i++ )
			computedOutput += weightedOutputs[i];
		
		return computedOutput;
	}
	
	public SOFNN( double[] initialVector )
	{
		double[] params = new double[qq.INPUT_VECTOR_LENGTH+1];
		// The first input vector determines the centers of the first neuron MFs.
		trainingIteration = 0;
		EBFNeurons.add( new EBFNeuron() );
		for( int i=0; i < qq.INPUT_VECTOR_LENGTH; i++ )
		{
			EBFNeurons.get( 0 ).mf[i].center = initialVector[i];
			EBFNeurons.get( 0 ).mf[i].width = qq.INITIAL_WIDTH;
		}
		
		for( int i=0; i < qq.INPUT_VECTOR_LENGTH+1; i++ )
			params[i] = 1;
		parameters.add( params );

		normalizationNeurons.add( new normalizationNeuron() );
		weightedNeurons.add( new weightedNeuron() );	
	}

	public double compute( double[] vector, Double desiredOutput ) 
	{
		if( trainingMode == true)
		{
			trainingIteration++;
			trainingPatterns.add( vector );
			desiredOutputs.add( desiredOutput );
		}
		double result = finalOutput( vector, desiredOutput );
        return result;
			
	}
}

