#conda install -c conda-forge numpy=1.20.0
import sys
import os
import multiprocessing
import time
import mandelbrot
import julia

# Hide pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

def checkArgs():
    if not len(sys.argv) == 3:
        print("Error: Invalid number of arguments. Arguments should be width, height")
        exit(1)

if __name__ == '__main__':
    print("Info: Program Start")
    print("Info: Recieved the following command line arguments: {}".format(sys.argv))
    
    # Check command line arguments
    checkArgs()

    # Create pipe
    mandelbrotMain, mandelbrotChild = multiprocessing.Pipe(True)
    juliaMain, juliaChild = multiprocessing.Pipe(True)

    # Setup processes
    mandelbrotProcess = multiprocessing.Process(
        target=mandelbrot.run, 
        args=(int(sys.argv[1]), int(sys.argv[2]), mandelbrotChild))

    juliaProcess = multiprocessing.Process(
        target=julia.run,
        args=(int(sys.argv[1]), int(sys.argv[2]), juliaChild))

    # Start processes
    print("Info: Starting processes")
    mandelbrotProcess.start()
    juliaProcess.start()

    while True:
        mandelbrotMain.send({"running":True}) # Tell mandelbrot process to keep going
        mandelbrotMsg = mandelbrotMain.recv() # Recieve status from mandelbrot process

        juliaMain.send({"running":True, "C":mandelbrotMsg["C"]}) # Tell julia process to keep going
        juliaMsg = juliaMain.recv() # Recieve status from mandelbrot process

        if not mandelbrotMsg["running"] or not juliaMsg["running"]:
            print("Info: One of the child processes quit")
            mandelbrotMain.send({"running":False})
            juliaMain.send({"running":False})
            break

    # Join processes
    print("Info: Joining processes")
    mandelbrotProcess.join()