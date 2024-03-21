class Person:
    def __init__(self, name):
        self.name = name

    def hello(self):
        print("Hello, " + self.name)

def main():
    person = Person('Gabriel')
    person.hello()

if __name__ == '__main__':
    main()