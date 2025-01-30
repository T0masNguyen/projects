#pragma once



using namespace std;
void Menu();
class Expenditure {
private:
	int monthlybudget, groceries, entertainment, rent, mobile, gym;
	string month;
public:
	Expenditure(string month, int rent, int mobile, int gym, int groceries, int entertainment, int monthlybudget);
	string getMonth();
	int getRent();
	int getMobile();
	int getGym();
	int getGroceries();
	int getEntertainment();
	int getMonthlyBudget();
};

class ExpenditureManager {
private:
	vector <Expenditure> Database;
public:
	void reading();
	void ViewAllExpenses();
	void MonthlyChecking();
	void ViewMonthlyBudget();

	void ViewRent();
	void ViewMobile();
	void ViewGym();
	void ViewGroceries();
	void ViewEntertainment();
};
