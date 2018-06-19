from cmd import Cmd
import traceback
import json, requests
import stela

st = stela.STELA()
send_url = 'http://freegeoip.net/json'
r = requests.get(send_url)
j = json.loads(r.text)
lat = j['latitude']
lon = j['longitude']
st.location = [lon, lat]
st.setup_cats()

class MyPrompt(Cmd):

    def default(self, line):
        try:
            dic = st.search(line)
            print dic["String"]
        except:
            traceback.print_exc()
    
    def do_details(self, line):
        print "\nShowing details..."
        print "lon: %.2f, lat: %.2f" % (lon, lat)
        print ("Objects loaded: \n"
               "\tMessier: %i\n"
               "\tHenry Draper: %i\n"
               "\tGJ: %i\n"
               "\tNGC: %i\n") % (len(st.catalogs.m), len(st.catalogs.hd),
                               len(st.catalogs.gj), len(st.catalogs.ngc))
    def do_EOF(self, line):
        return True

if __name__ == '__main__':
    prompt = MyPrompt()
    prompt.prompt = '>>> '
    prompt.cmdloop('Starting prompt...')
