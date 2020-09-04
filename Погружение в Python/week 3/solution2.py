import os
import csv

class CarBase:
    def __init__(self, brand, photo_file_name, carrying):
        self.brand = brand
        self.photo_file_name = photo_file_name
        self.carrying = float(carrying)
        
    def get_photo_file_ext(self):
        return os.path.splitext(self.photo_file_name)[1]


class Car(CarBase):
    def __init__(self, brand, photo_file_name, carrying, passenger_seats_count):
        super().__init__(brand, photo_file_name, carrying)
        self.car_type = 'car'
        self.passenger_seats_count = int(passenger_seats_count)


class Truck(CarBase):
    def __init__(self, brand, photo_file_name, carrying, body_whl):
        super().__init__(brand, photo_file_name, carrying)
        self.car_type = 'truck'
        self.body_whl = body_whl
        try:
            self.body_length, self.body_width, self.body_height = map(float, self.body_whl.split('x'))
        except (AttributeError, ValueError):
            self.body_length, self.body_width, self.body_height = 0.0, 0.0, 0.0
            
    def get_body_volume(self):
        return self.body_length * self.body_width * self.body_height


class SpecMachine(CarBase):
    def __init__(self, brand, photo_file_name, carrying, extra):
        super().__init__(brand, photo_file_name, carrying)
        self.car_type = 'spec_machine'
        self.extra = extra

def test_valid_text(param):
    if param == '':
        return False
    elif param == None:
        return False
    else:
        return True

def test_valid_photo(param):
    if param == '':
        return False
    elif param == None:
        return False
    elif os.path.splitext(param)[1] not in ['.jpg', '.jpeg', '.png', '.gif']:
        return False
    else:
        return True

def get_car_list(csv_filename):
    car_list = []
    
    with open(csv_filename) as csv_fd:
        reader = csv.reader(csv_fd, delimiter=';')
        next(reader)  # пропускаем заголовок
        for row in reader:
            if len(row) == 7:
                if (test_valid_text(row[1]) == False) or (test_valid_photo(row[3]) == False):
                    car = None
                else:
                    if row[0] == 'car':
                        try:
                            car = Car(row[1], row[3], row[5], row[2])
                        except ValueError:
                            car = None
                    elif row[0] == 'truck':
                        try:
                            car = Truck(row[1], row[3], row[5], row[4])
                        except ValueError:
                            car = None
                    elif row[0] == 'spec_machine':
                        if (test_valid_text(row[6]) == False):
                            car = None
                        else:
                            try:
                                car = SpecMachine(row[1], row[3], row[5], row[6])
                            except ValueError:
                                car = None
                    else:
                        car = None
                    
                if car:
                    car_list.append(car)
    return car_list