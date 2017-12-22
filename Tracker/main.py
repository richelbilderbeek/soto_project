__author__ = 'pieter'

import sys
import ConfigParser
import core

def main(args=None):
    """The main routine."""

    if not sys.argv[1:] is None:
        args = sys.argv[1:]

    #print("Arguments: ")
    #print(args)

    config = ConfigParser.ConfigParser()

    result = config.read(args[0])
    if not result:
        print "Error({0}): {1}".format("couldn't find file", args[0])
        return 0

    detector = core.Detector(config)
    # print "Start Tracking"
    detector.run()
    print "Finished Tracking"

    corrector = core.Correction(config)
    print "Start Correcting"
    corrector.run()
    print "Finished Correcting"

if __name__ == '__main__':
    main(['config.ini'])
