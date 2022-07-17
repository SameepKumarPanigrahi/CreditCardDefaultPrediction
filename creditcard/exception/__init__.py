from ast import Str
import os
import sys

class CreditCardException(Exception):
    def __init__(self, error_message:Exception, error_detail:sys):
        super().__init__(error_message)
        self.error_message = CreditCardException.get_detailed_message(error_message = error_message,
                                                                     error_detail = error_detail )
        
    @staticmethod
    def get_detailed_message( error_message:Exception, error_detail:sys)->Str:
        """
        error_message: Exception object
        error_detail: object of sys module
        """
        _,_,exec_tab = error_detail.exc_info()
        exception_block_line_number = exec_tab.tb_frame.f_lineno
        try_block_line_number = exec_tab.tb_lineno
        file_name = exec_tab.tb_frame.f_code.co_filename
        error_message = f"""
        Error occured in script: 
        [ {file_name} ] at 
        try block line number: [{try_block_line_number}] and exception block line number: [{exception_block_line_number}] 
        error message: [{error_message}]
        """
        return error_message
    
    def __str__(self):
        return self.error_message


    def __repr__(self) -> str:
        return CreditCardException.__name__.str()