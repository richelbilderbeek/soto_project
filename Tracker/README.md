# README #

## Richel's guide

Do

```
sudo ./install
```

to install everything.

Do 

```
./run
```

to run the project. If there is a warning it cannot find you movie file, modify `configuration_file.ini`.


## Old guid

This will start the tracking for each video file listed in config.ini and runs the correction at the end.
To use the program, type the following on the command line:

```
>> python main.py configuration_file.ini
```

Or just use `./run` from the command line. `run` assumes there is a video present in the same folder.

With the tester you can look at how the tracking performs over time. To use the tester, type the following on the command line:

```
>> python tester.py configuration_file.ini
```

 - `configuration_file.ini` can be used to change settings and video files.
 - Also you can have multiple config files, `config_mouse.ini`:

```
>> python main.py config_mouse.ini
```

## Errors

### `ImportError: No module named scipy`

Cause:

```
./run
```

Solution:

```
sudo pip install scipy
```

### `ImportError: No module named cv2`

Cause:

```
./run
```

Solution:

```
sudo pip install opencv-python
```

### `ImportError: No module named sklearn.cluster`

Cause:

```
./run
```

Solution:

```
sudo pip install -U scikit-learn
```

### `ImportError: No module named matplotlib.pyplot`

Cause:

```
./run
```

Solution:

```
sudo pip install matplotlib
```

### `ImportError: No module named _tkinter, please install the python-tk package`

Cause:

```
./run
```

Solution:

```
sudo apt-get install python-tk
```

### `AttributeError: 'module' object has no attribute 'FastFeatureDetector'`

Hypothesis: some OpenCV feature got changed, code needs to be modified for this.

## Contributers

 * Pieter Bosma
 * Richel Bilderbeek

## Richel's work

 * `install`: Linux only, installs all prerequisites
 * `run`: Linux only, one way to call the program