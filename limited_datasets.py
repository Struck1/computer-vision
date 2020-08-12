import os


def mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)


def link(src, dst):
    if not os.path.exists(dst):
         os.symlink(src, dst, target_is_directory=True)

mkdir('../data_files/fruits-360-small')



classes = [
  'Apple Golden 1',
  'Avocado',
  'Lemon',
  'Mango',
  'Kiwi',
  'Banana',
  'Strawberry',
  'Raspberry'
]

training_path_from = os.path.abspath('../data_files/fruits-360/Training')
test_path_from = os.path.abspath('../data_files/fruits-360/Test')

training_path_to = os.path.abspath('../data_files/fruits-360-small/Training')
test_path_to = os.path.abspath('../data_files/fruits-360-small/Test')

mkdir(training_path_to)
mkdir(test_path_to)

for c in classes:
    link(training_path_from + '/' + c, training_path_to + '/' + c )
    link(test_path_from + '/' + c, test_path_to + '/' + c)

