#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <string>
#include "Body.h"

void Menu() {
	ExpenditureManager Tom;
	Tom.reading();
	char Continue = 0;
	int option = 0;
	do {
		system("cls");
		cout << "+------------------------------+" << endl;
		cout << "|             Menu             |" << endl;
		cout << "+------------------------------+" << endl;
		cout << "|1.Search for expenses         |" << endl;
		cout << "+------------------------------+" << endl;
		cout << "|2.Checking monthly limit      |" << endl;
		cout << "+------------------------------+" << endl;
		cout << "|3.List all expenses           |" << endl;
		cout << "+------------------------------+" << endl;
		cout << "|4.Quit                        |" << endl;
		cout << "+------------------------------+" << endl;
		cout << "Enter your choice: ";
		cin >> option;

		switch (option) {
		case 1:
			system("cls");
			int c;
			cout << "+------------------------------+" << endl;
			cout << "|1.View monthly expenses       |" << endl;
			cout << "+------------------------------+" << endl;
			cout << "|2.View yearly expenses        |" << endl;
			cout << "+------------------------------+" << endl;
			cout << "|3.Return to the menu          |" << endl;
			cout << "+------------------------------+" << endl;
			cout << "Your choice: ";

			cin >> c;
			while (c != 3) {
				if (c == 1) {
					system("cls");
					string category;
					string months;
					cout << "+------------------------------------------------------------------------+" << endl;
					cout << "| Enter avaiable categories: Rent, Mobile, Gym, Groceries, Entertainment |" << endl;
					cout << "+------------------------------------------------------------------------+" << endl;
					cout << "Your choice: ";
					cin >> category;
					if (category == "Rent" || category == "rent") {
						system("cls");
						cout << "+---------------------+" << endl;
						cout << "| Rent from year 2020 |" << endl;
						cout << "+---------------------+" << endl;
						Tom.ViewRent();
						break;
					}
					else if (category == "Mobile" || category == "mobile") {
						system("cls");
						cout << "+-----------------------+" << endl;
						cout << "| Mobile fees from 2020 |" << endl;
						cout << "+-----------------------+" << endl << endl;
						Tom.ViewMobile();
						break;
					}
					else if (category == "Gym" || category == "gym") {
						system("cls");
						cout << "+--------------------+" << endl;
						cout << "| Gym fees from 2020 |" << endl;
						cout << "+--------------------+" << endl << endl;
						Tom.ViewGym();
						break;
					}
					else if (category == "Groceries" || category == "groceries") {
						system("cls");
						cout << "+------------------------------+" << endl;
						cout << "| Grocery each month from 2020 |" << endl;
						cout << "+------------------------------+" << endl << endl;
						Tom.ViewGroceries();
						break;
					}
					else if (category == "Entertainment" || category == "entertainment") {
						system("cls");
						cout << "+-------------------------------------------+" << endl;
						cout << "| Expense spent on entertainments from 2020 |" << endl;
						cout << "+-------------------------------------------+" << endl << endl;
						Tom.ViewEntertainment();
						break;
					}

					else cout << "Invalid category, try again!" << endl;


				}
				else if (c == 2) {
					system("cls");
					cout << "+---------------------------------------+" << endl;
					cout << "| Total spent from each month from 2020 |" << endl;
					cout << "+---------------------------------------+" << endl << endl;
					Tom.ViewMonthlyBudget();
					break;
				}
				break;
			}
			break;


		case 2:
			system("cls");
			/*Tom.ListAllExpenses();*/
			cout << "+---------------------------------------------+" << endl;
			cout << "|Checking monthly limit excess from 2020...   |" << endl;
			cout << "|Set limit per month is 650!                  |" << endl;
			cout << "+---------------------------------------------+" << endl << endl;
			Tom.MonthlyChecking();
			break;
		case 3:
			system("cls");
			cout << "+-----------------------------------------------------------------+" << endl;
			cout << "|              Listing all expenses in year 2020...               |" << endl;
			cout << "+-----------------------------------------------------------------+" << endl << endl;
			Tom.ViewAllExpenses();
			break;
		case 4:
			cout << endl;
			cout << "Program is terminated!" << endl;
			cout << endl;
			exit(0);
		default:
			cout << endl;
			cout << "Invalid choice! Try again" << endl;
			cout << endl;
		}

		cout << "\n" << "Would you like to continue: y/n" << endl;
		cin >> Continue;
	} while (Continue == 'y' || Continue == 'Y');
}