import sys
from pkg.logger import logging

def errorMessageDetail(err,errDetail:sys):
    _,_,exc_tb = errDetail.exc_info()
    fileName = exc_tb.tb_frame.f_code.co_filename
    lineNumber = exc_tb.tb_lineno
    errMsg = "Error occured in python script name[{0}] line number[{1}] error message[{2}]".format(fileName,lineNumber,str(err))
    return errMsg


class CustomException(Exception):
    def __init__(self, error_message, err_detail:sys):
        super().__init__(error_message)
        self.error_message = errorMessageDetail(error_message,errDetail=err_detail)

    def __str__(self):
        return self.error_message
    

if __name__ == "__main__":

    logging.info('Hello')
    print('HEllo')