class myclass():
    def __init__(self, params1):
        self.params = params1

    def print_param1(self):
        print(self.params)



if __name__ == "__main__":
    a = myclass(123)
    a.print_param1()
