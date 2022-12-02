## 1.1
## Check if "apple" is present in the fruits set.
fruits = {"apple", "banana", "cherry"}
if "apple" in fruits:
    print("Yes, apple is a fruit!")
    
## 1.2
## Use the add method to add "orange" to the fruits set.
fruits = {"apple", "banana", "cherry"}
fruits.add("orange")

## 1.3
## Use the correct method to add multiple items (more_fruits) to the fruits set.
fruits = {"apple", "banana", "cherry"}
more_fruits = ["orange", "mango", "grapes"]
fruits.update(more_fruits)

## 1.4
## Use the remove method to remove "banana" from the fruits set.    
fruits = {"apple", "banana", "cherry"}
fruits.remove("banana")

## 1.5
## Use the discard method to remove "banana" from the fruits set.
fruits = {"apple", "banana", "cherry"}
fruits.discard("banana")
