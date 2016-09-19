var test_lines = []
var train_lines = []


var fs = require('fs')

var lines = fs.readFileSync('./play.tsv').toString('utf8')
console.log(lines.length)
lines = lines.split('\r\n')

var play_keys = []

lines.forEach(function(l,idx){
  var o = convert_line(l)
  var text = strip_line_of_commas(o.text)
  var play = strip_line_of_commas(o.play)
  if(play_keys.indexOf(play) === -1){
    play_keys.push(play)
  }
  if(idx%2 === 0){
    test_lines.push([text,play_keys.indexOf(play)].join(','))
  } else {
    train_lines.push([text,play_keys.indexOf(play)].join(','))
  }
})

fs.writeFileSync('./train.csv', train_lines.join('\n'))
fs.writeFileSync('./test.csv', test_lines.join('\n'))

function convert_line(line){
  var l = line.split('\t')
  return {
    idx: l[0],
    play: l[1],
    sec1: l[2],
    sec2: l[3],
    who: l[4],
    text: l[5]
  }
}

function strip_line_of_commas(l){
  var r = ''
  for(var i = 0; i < l.length; i++){
    if(l.charAt(i) !== ','){
      r += l.charAt(i)
    }
  }
  return r
}
