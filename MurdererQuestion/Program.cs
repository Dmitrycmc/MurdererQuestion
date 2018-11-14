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
			var MurdererIsGray = Variable.Bernoulli(0.7).Named("MurdererIsGray");
			var WeaponIsRevolver = Variable.New<bool>().Named("WeaponIsRevolver");
			var FoundGraysHair = Variable.New<bool>().Named("FoundGraysHair");
			var Found2ndHair = Variable.New<bool>().Named("Found2ndHair");

			using (Variable.If(MurdererIsGray))
			{
				WeaponIsRevolver.SetTo(Variable.Bernoulli(0.9));
				FoundGraysHair.SetTo(Variable.Bernoulli(0.5));
				Found2ndHair.SetTo(Variable.Bernoulli(0.05));
			}

			using (Variable.IfNot(MurdererIsGray))
			{
				WeaponIsRevolver.SetTo(Variable.Bernoulli(0.2));
				FoundGraysHair.SetTo(Variable.Bernoulli(0.05));
				Found2ndHair.SetTo(Variable.Bernoulli(0.5));
			}

			InferenceEngine engine = new InferenceEngine();
			WeaponIsRevolver.ObservedValue = false;
			FoundGraysHair.ObservedValue = false;
			Found2ndHair.ObservedValue = true;
			InferenceEngine.Visualizer = new Microsoft.ML.Probabilistic.Compiler.Visualizers.WindowsVisualizer();
			engine.ShowFactorGraph = true;
			Console.WriteLine("MurdererIsGray: " + engine.Infer(MurdererIsGray));
			
			/// 1 фактор граф
			/// 2 продемострировать примеры
			/// 3 написать линейную регрессию
		}
	}
}
