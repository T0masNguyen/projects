#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <string>
#include "Body.h"


using namespace std;


Expenditure::Expenditure(string month, int rent, int mobile, int gym, int groceries, int entertainment, int monthlybudget) {
	this->month = month;
	this->rent = rent;
	this->mobile = mobile;
	this->gym = gym;
	this->groceries = groceries;
	this->entertainment = entertainment;
	this->monthlybudget = monthlybudget;
}
string Expenditure::getMonth() {
	return month;
}
int Expenditure::getRent() {
	return rent;
}
int Expenditure::getMobile() {
	return mobile;
}
int Expenditure::getGym() {
	return gym;
}
int Expenditure::getGroceries() {
	return groceries;
}
int Expenditure::getEntertainment() {
	return entertainment;
}
int Expenditure::getMonthlyBudget() {
	return monthlybudget;
}



void ExpenditureManager::reading() {
	string Srent, Sgroceries, Sgym, Smobile, Sentertainments, Smonth, Sname, Scategory, Smonthlybudget, line;
	ifstream file;
	file.open("data.csv", ios::in);
	if (file.fail())
	{
		cout << "File was not found\n";
		system("PAUSE");
		exit(1);
	}

	// skip first line to separate data easier
	getline(file, line);
	// getting data from file
	while (getline(file, line))
	{
		vector <int> expense;
		vector <string> time;
		stringstream data(line);
		getline(data, Smonth, ',');
		time.push_back(Smonth);
		getline(data, Srent, ',');
		expense.push_back(stoi(Srent));
		getline(data, Smobile, ',');
		expense.push_back(stoi(Smobile));
		getline(data, Sgym, ',');
		expense.push_back(stoi(Sgym));
		getline(data, Sgroceries, ',');
		expense.push_back(stoi(Sgroceries));
		getline(data, Sentertainments, ',');
		expense.push_back(stoi(Sentertainments));
		getline(data, Smonthlybudget, ',');
		expense.push_back(stoi(Smonthlybudget));
		Expenditure Tom(time[0], expense[0], expense[1], expense[2], expense[3], expense[4], expense[5]);
		Database.push_back(Tom);
	}
}
void ExpenditureManager::ViewAllExpenses() {
	ifstream file;
	file.open("data.csv", ios::in);
	string Srent, Sgroceries, Sgym, Smobile, Sentertainments, Smonth, Sname, Scategory, Smonthlybudget, line;
	while (getline(file, line)) {
		stringstream data(line);
		getline(data, Smonth, ',');
		getline(data, Srent, ',');
		getline(data, Smobile, ',');
		getline(data, Sgym, ',');
		getline(data, Sgroceries, ',');
		getline(data, Sentertainments, ',');
		cout << Smonth << setw(22) << "\t" << Srent << setw(10) << Smobile << setw(8) << Sgym << setw(15) << Sgroceries << setw(18) << Sentertainments << endl;
		cout << "------------------------------------------------------------------------------------------------" << endl;
	}
}
void ExpenditureManager::MonthlyChecking() {
	for (size_t i = 0; i < Database.size(); i++)
	{
		cout << Database[i].getMonth();
		cout << ":" << setw(22) << "\t";
		if (Database[i].getMonthlyBudget() > 650) {
			cout << "Exceeded limit!" << endl;
		}
		else cout << "Within Limit!" << endl;
		cout << "-----------------------------------------------" << endl;


	}
}
void ExpenditureManager::ViewMonthlyBudget() {
	for (size_t i = 0; i < Database.size(); i++)
	{
		cout << Database[i].getMonth() << setw(15) << "\t";
		cout << Database[i].getMonthlyBudget() << endl;
		cout << "---------------------------" << endl;
	}
}
void ExpenditureManager::ViewRent() {
	for (size_t i = 0; i < Database.size(); i++)
	{
		cout << Database[i].getMonth() << setw(15) << "\t";
		cout << Database[i].getRent() << endl;
		cout << "---------------------------" << endl;
	}
}
void ExpenditureManager::ViewMobile() {
	for (size_t i = 0; i < Database.size(); i++)
	{
		cout << Database[i].getMonth() << setw(15) << "\t";
		cout << Database[i].getMobile() << endl;
		cout << "---------------------------" << endl;
	}
}
void ExpenditureManager::ViewGym() {
	for (size_t i = 0; i < Database.size(); i++)
	{
		cout << Database[i].getMonth() << setw(15) << "\t";
		cout << Database[i].getGym() << endl;
		cout << "---------------------------" << endl;
	}
}
void ExpenditureManager::ViewGroceries() {
	for (size_t i = 0; i < Database.size(); i++)
	{
		cout << Database[i].getMonth() << setw(15) << "\t";
		cout << Database[i].getGroceries() << endl;
		cout << "---------------------------" << endl;
	}
}
void ExpenditureManager::ViewEntertainment() {
	for (size_t i = 0; i < Database.size(); i++)
	{
		cout << Database[i].getMonth() << setw(15) << "\t";
		cout << Database[i].getEntertainment() << endl;
		cout << "---------------------------" << endl;
	}
}
