import os
import argparse

def get_parser() -> argparse.ArgumentParser:
    '''
    :return: an argparse ArgumentParser object for parsing command
        line parameters
    '''
    parser = argparse.ArgumentParser(description='Complete pre-processing and script generator for CGPGRN.', epilog = 'Example usage to process data: python completePreProcessing.py -cm KMeans -nc 10 -e 500nTF-ExpressionData.csv -p PseudoTime.csv -pn hHep -s 500nTF -d BiKMeans -r 5')

    parser.add_argument('-o', dest = 'MAXOUT', type = str,
                        default = 'None',
                        help='Maximum number of outputs \n'
                        )

    parser.add_argument('-n', dest = 'MAXNODE', type = str,
                        default = 'None',
                        help='Maximum number of nodes \n'
                        )

    parser.add_argument('-e', dest = 'MAXEVAL', type = str,
                        default = 'None',
                        help='Maximum number of evaluations \n'
                        )      


    return parser                        


def parse_arguments():
    '''
    Initialize a parser and use it to parse the command line arguments
    :return: parsed dictionary of command line arguments
    '''
    parser = get_parser()
    opts = parser.parse_args()

    return opts


def make(MAX_OUTPUTS, MAX_NODES, MAX_EVAL):

    makeFileName = 'Makefile'
    makeFileOpen = open(makeFileName, 'w')
    makeFileOpen.write("CC = g++\n")
    makeFileOpen.write("CFLAGS = -g -Wall\n\n")
    makeFileOpen.write("MAXOUTVALUE = " + str(MAX_OUTPUTS) + "\n")
    makeFileOpen.write("MAXNODESVALUE = " + str(MAX_NODES) + "\n")
    makeFileOpen.write("MAXEVAL = " + str(MAX_EVAL) + "\n\n")
    makeFileOpen.write("main:\n")
    if os.name == 'nt':
        makeFileOpen.write('\t${CC} -DMAX_OUTPUTS=${MAXOUTVALUE} -DMAX_NODES=${MAXNODESVALUE} -DNUM_GENERATIONS=${MAXEVAL} source/*.cpp* -o progW -I"source/amd/include" -lOpenCl -L"source/amd/lib/x86_64"\n')
    else:
        makeFileOpen.write('\t${CC} -DMAX_OUTPUTS=${MAXOUTVALUE} -DMAX_NODES=${MAXNODESVALUE} -DNUM_GENERATIONS=${MAXEVAL} source/*.cpp* -o progL -lOpenCL\n')
    makeFileOpen.close()


    #makeFileName = 'Makefile_org'
    #makeFileOpen = open(makeFileName, 'r')

if __name__ == '__main__':
    opts = parse_arguments()
    MAXOUT = opts.MAXOUT
    MAXNODE = opts.MAXNODE
    MAXEVAL = opts.MAXEVAL
    make(MAXOUT, MAXNODE, MAXEVAL)


    if os.name != 'nt':
        os.system('chmod +x Makefile')
        os.system('make')
    else:
        os.system('cmd /c "make"')    
