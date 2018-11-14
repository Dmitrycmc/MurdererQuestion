using Microsoft.ML.Probabilistic.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MurdererQuestion
{
	class Program
	{
		static void Main(string[] args)
		{
			var MurdererIsGray = Variable.Bernoulli(0.3);
			var WeaponIsRevolver = Variable.New<bool>();
			var FoundGraysHair = Variable.New<bool>();

			using (Variable.If(MurdererIsGray))
			{
				WeaponIsRevolver = Variable.Bernoulli(0.9);
				FoundGraysHair = Variable.Bernoulli(0.5); 
			}

			using (Variable.IfNot(MurdererIsGray))
			{
				WeaponIsRevolver = Variable.Bernoulli(0.2);
				FoundGraysHair = Variable.Bernoulli(0.05);
			}

			InferenceEngine engine = new InferenceEngine();
			WeaponIsRevolver.ObservedValue = true;
			FoundGraysHair.ObservedValue = true;
			Console.WriteLine("MurdererIsGray: " + engine.Infer(MurdererIsGray));

		}
	}
}
