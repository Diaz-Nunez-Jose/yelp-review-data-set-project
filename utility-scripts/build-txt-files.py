import pandas as pd

# unique_categories = {"Nightlife", "Lounges", "Venues & Event Spaces", "Active Life", "Bowling", "Grocery", 
# 					 "Meat Shops", "Dance Clubs", "Music Venues", "Karaoke", "Shopping Centers", "Shopping", 
# 					 "Outlet Stores", "Convenience Stores", "Hotels & Travel", "Hotels", "Jazz & Blues", 
# 					 "Cinema", "Pool Halls", "Arcades", "Casinos", "Health Markets", "Social Clubs", 
# 					 "Food Delivery Services", "Hookah Bars", "Amusement Parks", "Gas & Service Stations",
# 					 "Do-It-Yourself Food", "Cafeteria", "Wineries", "Bed & Breakfast", 
# 					 "Landmarks & Historical Buildings", "Street Vendors", "Festivals", "Farmers Market",
# 					 "Butcher", "Country Dance Halls", "Cultural Center", "Delicatessen", "Food", "Fast Food", 
# 					 "Bars", "Bakeries", "Coffee & Tea", "Donuts", "Caterers", "Dive Bars", "Pubs", "Buffets", 
# 					 "Cafes", "Sports Bars", "Specialty Food", "Gluten-Free", "Wine Bars", "Comfort Food", 
# 					 "Bagels", "Gastropubs", "Juice Bars & Smoothies", "Breweries", "Pretzels", "Food Stands", 
# 					 "Island Pub", "Tapas Bars", "Cheese Shops", "Gay Bars", "Herbs & Spices", "Hot Pot", 
# 					 "Local Flavor", "Brasseries", "Shaved Ice", "Food Trucks", "Food Court", "Champagne Bars", 
# 					 "Bubble Tea", "Piano Bars", "Poutineries", "Beer Bar", "Distilleries", "Lebanese", 
# 					 "Soup", "Caribbean", "Tea Rooms", "Cheesesteaks", "Soul Food", "Salvadoran", "Kosher", 
# 					 "Polish", "Creperies", "Cuban", "Russian", "Irish", "Fruits & Veggies", "Fondue", 
# 					 "Arabian", "Seafood Markets", "Peruvian", "Halal", "Dim Sum", "Mongolian", 
# 					 "Persian/Iranian", "German", "Cantonese", "Taiwanese", "Argentine", 
# 					 "Himalayan/Nepalese", "Moroccan", "Falafel", "Ethiopian", "African", "Indonesian", 
# 					 "Turkish", "Afghan", "Tapas/Small Plates", "Basque", "Spanish", "Cocktail Bars", 
# 					 "Brazilian", "Personal Chefs", "Laotian", "Szechuan", "Belgian", "Gelato", 
# 					 "Live/Raw Food", "Bistros", "Chocolatiers & Shops", "Malaysian", "Singaporean", 
# 					 "Burmese", "Scandinavian", "Canadian (New)", "Czech", "Slovakian", "Scottish", 
# 					 "Modern European", "Bangladeshi", "Ramen", "Portuguese", "Ukrainian", "Shanghainese", 
# 					 "Cambodian", "Venezuelan", "Colombian", "Dominican", "Patisserie/Cake Shop", 
# 					 "Australian", "Egyptian"}

unique_categories = {"Do-It-Yourself Food", "Fast Food", "Bakeries", "Buffets", "Comfort Food", "Food Stands", 
					"Island Pub", "Tapas Bars", "Cheese Shops", "Herbs & Spices", "Hot Pot", "Local Flavor", 
					"Brasseries", "Food Trucks", "Food Court", "Lebanese", "Soup", "Caribbean", "Tea Rooms", 
					"Cheesesteaks", "Soul Food", "Salvadoran", "Kosher", "Polish", "Creperies", "Cuban", 
					"Russian", "Irish", "Fruits & Veggies", "Fondue", "Arabian", "Seafood Markets", "Peruvian", 
					"Halal", "Dim Sum", "Mongolian", "Persian/Iranian", "German", "Cantonese", "Taiwanese", 
					"Argentine", "Himalayan/Nepalese", "Moroccan", "Falafel", "Ethiopian", "African", "Indonesian", 
					"Turkish", "Afghan", "Tapas/Small Plates", "Basque", "Spanish", "Laotian", "Szechuan", 
					"Belgian", "Gelato", "Live/Raw Food",  "Malaysian", "Singaporean", "Burmese", "Scandinavian", 
					"Canadian (New)", "Czech", "Slovakian", "Scottish", "Modern European", "Bangladeshi", "Ramen", 
					"Portuguese", "Ukrainian", "Shanghainese", "Cambodian", "Venezuelan", "Colombian", "Dominican",
					"Australian", "Egyptian"}

df = pd.read_csv('yelp_data_set/yelp_review_business_joined.csv')

for row in df.itertuples():
	if not isinstance(row.categories, float) and isinstance(row.categories, str):
		text = row.text
		categories = [cats for cats in row.categories.strip().split(';')]
		for category in categories:
			if category in unique_categories:
				f = open("category-txt-files/" + category.replace("/", "-") + ".txt", "a+")
				# f.write(text + " ", encoding="UTF-8")
				f.write(str(text.encode('utf8') + b" "))
				f.close()