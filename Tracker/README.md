# README #

This will start the tracking for each video file listed in config.ini and runs the correction at the end.
To use the program, type the following on the command line:

```
>> python main.py config.ini
```

With the tester you can look at how the tracking performs over time. To use the tester, type the following on the command line:

```
>> python tester.py config.ini
```

- config.ini can be used to change settings and video files.
- Also you can have multiple config files, config_mouse.ini:

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

## Contributers

 * Pieter Bosma
 * Richel Bilderbeek

## Richel's work

 * `install`: Linux only, installs all prerequisites
 * `run`: Linux only, one way to call the program