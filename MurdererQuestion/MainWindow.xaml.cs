using Microsoft.ML.Probabilistic.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MurdererQuestion
{
	/// <summary>
	/// Логика взаимодействия для MainWindow.xaml
	/// </summary>
	public partial class MainWindow : Window
	{
		public double initProb { get; set; } = 0.5;
		public ObservedBool weaponIsRevolver { get; set; } = ObservedBool.Undef;
		public ObservedBool foundMansHair { get; set; } = ObservedBool.Undef;
		public ObservedBool foundWomansHair { get; set; } = ObservedBool.Undef;

		public MainWindow()
		{
			InitializeComponent();
			Model.Init(initProb);
		}

		private void button_Click(object sender, RoutedEventArgs e)
		{
			try
			{
				string ans = Model.Exec(weaponIsRevolver, foundMansHair, foundWomansHair);
				MessageBox.Show(ans);

			} catch (Exception exception)
			{
				MessageBox.Show(exception.Message);
			}
		}

		private void TextBox_TextChanged(object sender, TextChangedEventArgs e)
		{
			Model.Init(initProb);
		}
	}
}
