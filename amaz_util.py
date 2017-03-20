import urllib.request
import tarfile
from os import system
import sys
import pickle

class Utility(object):

    def __init__(self):
        self.name = "util"

    def download_file_fromurl(self,url,destination,header='--header="Accept: text/html" --user-agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0" '):
        cmd = 'wget ' + header + url + ' -O ' + destination + ' -q'
        system(cmd)

    def untar_file(self,filepath):
        if filepath.endswith(".gz"):
            cmd = 'tar -zxvf ' + filepath + ' -C ./'
            system(cmd)
        else:
            print("the given filepath was not tar.gz")

    def unpickle(self,filepath):
        fp = open(filepath, 'rb')
        if sys.version_info.major == 2:
            data = pickle.load(fp)
        elif sys.version_info.major == 3:
            data = pickle.load(fp, encoding='latin-1')
        fp.close()
        return data

    def savepickle(self,data,savepath):
        with open(savepath,'wb') as f:
            pickle.dump(data,f)
        return True
