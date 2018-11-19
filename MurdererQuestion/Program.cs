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
			Console.Write("Enter init P (Murderer is Man): ");
			Model.Init(double.Parse(Console.ReadLine().Replace('.', ',')));

			for (int i = 0; i < 27; i++)
			{
				int a = i % 3;
				int b = i / 3 % 3;
				int c = i / 9 % 3;
				if (b == c) continue;

				Model.Exec((ObservedBool)a, (ObservedBool)b, (ObservedBool)c);

			}
			
			/// 1 фактор граф
			/// 2 продемострировать примеры
			/// 3 написать линейную регрессию
		}
	}
}
