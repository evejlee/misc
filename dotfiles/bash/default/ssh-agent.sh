res=`which keychain 2>&1> /dev/null`
if [ $? == 0 ]; then
	keychain --noinherit id_rsa 2>/dev/null
	. ~/.keychain/${HOSTNAME}-sh > /dev/null
fi

