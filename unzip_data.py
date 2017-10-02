import zipfile
import os


def main():
    for item in os.listdir('.'):
        # Check for dirs
        if os.path.isdir(item) and not item.startswith('.'):
            os_dir = item
            # Check if the dir has a 'data.zip' file:
            data_filename = os.path.join(os_dir, 'data.zip')
            if os.path.exists(data_filename):                
                with zipfile.ZipFile(data_filename) as data_zipfile:
                    print('Extracting {}'.format(data_filename))
                    data_zipfile.extractall(os_dir)

if __name__ == '__main__':
    main()