# coding=utf-8

from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib

user_name = "592191580@qq.com"
passwd = "@thzy1994"


def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr(( \
        Header(name, 'utf-8').encode(), \
        addr.encode('utf-8') if isinstance(addr, unicode) else addr))


def send_email(text):
    smtpObj = smtplib.SMTP("smtp.qq.com", 25)

    msg = MIMEText(text, 'plain', 'utf-8')
    msg['From'] = _format_addr(u'我的python程序 <%s>' % user_name)
    msg['To'] = _format_addr(u'周炀')
    msg['Subject'] = Header(u'python通知内容', 'utf-8').encode()

    # smtpObj.set_debuglevel(1)
    smtpObj.login(user_name, passwd)
    smtpObj.sendmail(user_name, user_name, msg.as_string())
    smtpObj.quit()


if __name__ == "__main__":
    send_email("hello world from zy")
