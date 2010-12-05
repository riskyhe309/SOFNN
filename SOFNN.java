package craig;

// Robin Gane-McCalla 11.14.10
// This is a prototype of a Self Organizing Neural Network as described in
//An on-line algorithm for creating self-organizing fuzzy neural networks
//(Leng, Prasad, McGinnity 2004)
//it implements the five layers described in that paper but at present
//doesn't learn or add or remove neurons.  Most of the information
//in here comes from page 4 of the paper.

// Modified by craig 11/28/10

import java.util.ArrayList; 

public class SOFNN 
{
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

	//EBFNeurons are the first layer after the input
	class EBFNeuron 
	{
		//an EBF neuron will have multiple membershipFunctions which take
		//values from the input to compute the output of the EBF neuron
		public ArrayList<membershipFunction> mf = new ArrayList<membershipFunction>();

		public EBFNeuron()
		{
			for( int i=0; i < qq.INPUT_VECTOR_LENGTH; i++ )
			{
				membershipFunction newMF = new membershipFunction();
				newMF.center = initialCenters[i];
				newMF.width = qq.INITIAL_WIDTH;
				mf.add( newMF);
			}
		}
		
		public double output( double input[] )
		{
			double tempOutput = 0;
			double output;

			//The output of an EBF neuron is the negated sum of its membership functions
			//to the eth power.
			//We subtract the center from the input and square that number
			//then divide the result by two times the width squared, just like in the paper.
			for( int i=0; i < mf.size(); i++ )
				tempOutput += Math.pow( ( input[i] - mf.get( i ).center ), 2.0 ) / ( 2 * Math.pow( mf.get( i ).width, 2.0 ) );			
			output = Math.exp( -tempOutput );
			return output;
		}
	}
	
	//the normalization layer makes the sum of all the neurons equal to one
	//with each one weighted according to the percentage of the total output it has
	class normalizationNeuron
	{		
		public normalizationNeuron()
		{
				
		}
		
		double output( double input, double sum )
		{
			double normalized;
			//here we return the output of a specific neuron divided by the 
			//sum of all the neurons
			normalized = input/sum;
			return normalized;
		}		
	}

	//the weighted layer multiplies all the outputs by a bias value
	class weightedNeuron
	{
		public double weight=11181.23;
		
		weightedNeuron()
		{
			
		}
		
		double output( double input )
		{
			double weighted = weight*input;
			return weighted;
		}
	}
	
	double finalOutput( double[] vector )
	{
		double computedOutput = 0;
		double error;
		double EBFSum = 0;
		double[] EBFOutputs = new double[EBFNeurons.size()];
		double[] normalizedOutputs = new double[normalizationNeurons.size()];
		double[] weightedOutputs = new double[weightedNeurons.size()];
		double[][][] dist = new double[vector.length][EBFNeurons.size()][EBFNeurons.size()];
		boolean firingStrengthSatisfied = false;
		
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
				firingStrengthSatisfied = true;			
		}

		error = Math.abs( computedOutput - expectedOutput );
		
		// Criterion A
		// No adjustment needed
		if( (  error <= qq.ERROR_TOLERANCE ) && ( firingStrengthSatisfied == true ) )
		{

		}
		
		// Criterion B
		// Width needs adjustment
		if( ( error <= qq.ERROR_TOLERANCE ) && ( firingStrengthSatisfied == false ) )
		{
			
		}
		
		// Criterion C
		// New neuron needed
		if( ( error > qq.ERROR_TOLERANCE ) && ( firingStrengthSatisfied == true ) )
		{
	        for( int i=0; i < vector.length; i++)
	        {
	        	for( int j=0; j < EBFNeurons.size(); j++  )
	        	{
	        		for( int k=0; k < EBFNeurons.get( i ).mf.size(); k++ )
	        			dist[i][j][k] = Math.abs( vector[i] - EBFNeurons.get( j ).mf.get( k ).center );	        		
	        	}
	        }
		}
		
		// Criterion D
		// New neuron needed
		if( ( error > qq.ERROR_TOLERANCE ) && ( firingStrengthSatisfied == false ) )
		{

		}		

		return computedOutput;
	}
	
	public SOFNN()
	{
		initialCenters[0] = 1;
		initialCenters[1] = 1;
		initialCenters[2] = 1;
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

