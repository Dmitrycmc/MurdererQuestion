using System;
using Microsoft.ML.Probabilistic.Models;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MurdererQuestion
{
	public enum ObservedBool { True, False, Undef };

	public static class Model
	{
		static Variable<bool> MurdererIsMan;
		static Variable<bool> WeaponIsRevolver;
		static Variable<bool> FoundMansHair;
		static Variable<bool> FoundWomansHair;
		static InferenceEngine engine = null;

		public static void Init(
			double initProb = 0.3,
			double weaponIsRevIfMurdererIsMan = 0.9,
			double weaponIsRevIfMurdererIsNotMan = 0.2,
			double foundMansHairIfMurdererIsMan = 0.5,
			double foundMansHairIfMurdererIsNotMan = 0.05,
			double foundWomansHairIfMurdererIsMan = 0.05,
			double foundWomansHairIfMurdererIsNotMan = 0.5
		) {
			MurdererIsMan = Variable.Bernoulli(initProb).Named("MurdererIsMan");
			WeaponIsRevolver = Variable.New<bool>().Named("WeaponIsRevolver");
			FoundMansHair = Variable.New<bool>().Named("FoundMansHair");
			FoundWomansHair = Variable.New<bool>().Named("FoundWomansHair");

			using (Variable.If(MurdererIsMan))
			{
				WeaponIsRevolver.SetTo(Variable.Bernoulli(weaponIsRevIfMurdererIsMan));
				FoundMansHair.SetTo(Variable.Bernoulli(foundMansHairIfMurdererIsMan));
				FoundWomansHair.SetTo(Variable.Bernoulli(foundWomansHairIfMurdererIsMan));
			}

			using (Variable.IfNot(MurdererIsMan))
			{
				WeaponIsRevolver.SetTo(Variable.Bernoulli(weaponIsRevIfMurdererIsNotMan));
				FoundMansHair.SetTo(Variable.Bernoulli(foundMansHairIfMurdererIsNotMan));
				FoundWomansHair.SetTo(Variable.Bernoulli(foundWomansHairIfMurdererIsNotMan));
			}

			engine = new InferenceEngine();
		}

		private static void set(Variable<bool> variable, ObservedBool observedBool)
		{
			if (observedBool == ObservedBool.Undef)
			{
				variable.ClearObservedValue();
				return;
			}
			variable.ObservedValue = observedBool == ObservedBool.True;
		}
		
		public static string Exec(
			ObservedBool WeaponIsRevolver = ObservedBool.True, 
			ObservedBool FoundMansHair = ObservedBool.True, 
			ObservedBool FoundWomansHair = ObservedBool.Undef,
			bool showGraph = false
		) {
			if (engine == null)
			{
				throw new Exception("Uninitialed engine!");
			}

			set(Model.WeaponIsRevolver, WeaponIsRevolver);
			set(Model.FoundMansHair, FoundMansHair);
			set(Model.FoundWomansHair, FoundWomansHair);

			if (showGraph)
			{
				InferenceEngine.Visualizer = new Microsoft.ML.Probabilistic.Compiler.Visualizers.WindowsVisualizer();
				engine.ShowFactorGraph = true;
			}

			string weapon;
			string mansHair;
			string womansHair;

			if (WeaponIsRevolver == ObservedBool.True) weapon = "revolver";
			else if (WeaponIsRevolver == ObservedBool.False) weapon = "knife";
			else weapon = "no data";

			if (FoundMansHair == ObservedBool.True) mansHair = "found!";
			else if (FoundMansHair == ObservedBool.False) mansHair = "not found!";
			else mansHair = "no data";
			
			if (FoundWomansHair == ObservedBool.True) womansHair = "found!";
			else if (FoundWomansHair == ObservedBool.False) womansHair = "not found!";
			else womansHair = "no data";

			string res = engine.Infer(MurdererIsMan).ToString();
			double probMan = Math.Round(100 * double.Parse(res.Substring(10, res.Length - 1 - 10)), 2);
			double probWoman = 100 - probMan;

			string ans = "Weapon: " + weapon + Environment.NewLine;
			ans += "Man's hair: " + mansHair + Environment.NewLine;
			ans += "Woman's hair: " + womansHair + Environment.NewLine;
			ans += "Murderer is man: " + probMan + " %" + Environment.NewLine;
			ans += "Murderer is woman: " + probWoman + " %" + Environment.NewLine;
	
			return ans;
		}
	}
}
