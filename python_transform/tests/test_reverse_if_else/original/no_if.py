def greet(language):
    greetings = {
        "en": "Hello",
        "es": "Hola",
        "fr": "Bonjour"
    }
    return greetings.get(language, "Hello")

numbers = [1, 2, 3, 4, 5]
even_numbers = [num for num in numbers if num % 2 == 0]

print(even_numbers)  
print(greet("es"))