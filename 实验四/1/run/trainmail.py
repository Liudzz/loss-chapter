import smtplib
from email.mime.text import MIMEText


def send_mail(content,subject):
    mail_from = ""
    passwd = ""
    mail_to = ""

    # subject = "Train Report"
    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = mail_from
    msg['To'] = mail_to

    try:
        s = smtplib.SMTP_SSL("",465)
        s.login(mail_from,passwd)
        s.sendmail(mail_from,mail_to,msg.as_string())
    except :
            print('Connect Error')
