#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests the audacity pipe.
Keep pipe_test.py short!!
You can make more complicated longer tests to test other functionality
or to generate screenshots etc in other scripts.
Make sure Audacity is running first and that mod-script-pipe is enabled
before running this script.
Requires Python 2.7 or later. Python 3 is strongly recommended.
"""

import os
import time
import sys


if sys.platform == 'win32':
    print("pipe-test.py, running on windows")
    TONAME = '\\\\.\\pipe\\ToSrvPipe'
    FROMNAME = '\\\\.\\pipe\\FromSrvPipe'
    EOL = '\r\n\0'
else:
    print("pipe-test.py, running on linux or mac")
    TONAME = '/tmp/audacity_script_pipe.to.' + str(os.getuid())
    FROMNAME = '/tmp/audacity_script_pipe.from.' + str(os.getuid())
    EOL = '\n'

print("Write to  \"" + TONAME +"\"")
if not os.path.exists(TONAME):
    print(" ..does not exist.  Ensure Audacity is running with mod-script-pipe.")
    sys.exit()

print("Read from \"" + FROMNAME +"\"")
if not os.path.exists(FROMNAME):
    print(" ..does not exist.  Ensure Audacity is running with mod-script-pipe.")
    sys.exit()

print("-- Both pipes exist.  Good.")

TOFILE = open(TONAME, 'w')
print("-- File to write to has been opened")
FROMFILE = open(FROMNAME, 'rt')
print("-- File to read from has now been opened too\r\n")


def send_command(command):
    """Send a single command."""
    print("Send: >>> \n"+command)
    TOFILE.write(command + EOL)
    TOFILE.flush()

def get_response():
    """Return the command response."""
    result = ''
    line = ''
    while line != '\n':
        result += line
        line = FROMFILE.readline()
        #print(" I read line:["+line+"]")
    return result

def do_command(command):
    """Send one command, and return the response."""
    send_command(command)
    response = get_response()
    print("Rcvd: <<< \n" + response)
    return response

def quick_test():
    """Example list of commands."""
    do_command('Help: Command=Help')
    do_command('Help: Command="GetInfo"')
    #do_command('SetPreference: Name=GUI/Theme Value=classic Reload=1')

def devil_loc(filepath):
	for release in [0]:
		for crunch in [.1, .33, .66, .84]:
			for crush in [.1, .33, .66, .91]:
				for darkness in [.1, .33, .66, .85]:
					do_command('Import2: Filename=' + filepath)
					do_command('SelectAll:')
					filename = "devilloc" + "_release=" + str(release) + "_crunch=" + str(crunch) + "_crush=" + str(crush) + "_darkness=" + str(darkness) + ".wav" 
					do_command('Devil-LocDeluxe: Bypass=0 Crunch='+str(crunch)+' Crush='+str(crush)+' Darkness='+str(darkness)+' Mix=1 Release='+str(release))
					do_command('Export2: Filename=/Users/maxmines/Documents/DSP/AIGear/devil_loc_deluxe_train' + filename + ' NumChannels=1')
					time.sleep(10)
					do_command('SelectAll:')
					do_command('RemoveTracks:')

def supermassive(filepath):
	for delayms in [0.33, .5, .77]:
		for delaywarp in [.33, .5, .79]:
			do_command('Import2: Filename=' + filepath)
			do_command('SelectAll:')
			filename = "supermassive" + "_delayms=" + str(delayms) + "_delaywarp=" + str(delaywarp) + ".wav" 
			do_command('ValhallaSupermassive: Clear=1 Delay_Ms=' + str(delayms) + ' DelayNote=0.59200001 DelaySync=0.25 DelayWarp='+ str(delaywarp) + ' Density=1 Feedback=0.75 HighCut=0.189 LowCut=0 Mix=1 ModDepth=0.5 Mode=0.33333334 ModRate=0.27383411 Reserved1=0 Reserved2=0 Reserved3=0 Reserved4=0 Width=1')
			do_command('Export2: Filename=/Users/maxmines/Documents/DSP/AIGear/Valhalla_Supermassive/test/' + filename + ' NumChannels=1')
			time.sleep(10)
			do_command('SelectAll:')
			do_command('RemoveTracks:')

supermassive('/Users/maxmines/Documents/DSP/AIGear/Valhalla_Supermassive/test/supermassive_clean.wav')