from creditcard.logger import logging
from creditcard.exception import CreditCardException
from flask import Flask

import sys, os

app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        #raise Exception('Custom Exception')
        return ("Reaching here") 
    except Exception as e:
        raise CreditCardException(e, sys) from e

if __name__ == '__main__':
    app.run(debug=True)