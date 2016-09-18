var fs = require('fs')
var crypto = require('crypto')

var n_elements = 10000

console.log(make_bytes(true))
console.log(make_bytes(false))

var csv_lines = []
for(var i = 0; i < n_elements; i++){
  if(i%2 === 0){
    // full range
    csv_lines.push([make_bytes(true),'yes'].join(','))
  } else {
    // restricted
    csv_lines.push([make_bytes(false),'no'].join(','))
  }
}

fs.writeFileSync('./train.csv', csv_lines.slice(0,n_elements/2).join('\n'))
fs.writeFileSync('./test.csv', csv_lines.slice(n_elements/2).join('\n'))

function make_bytes(restricted){
  var l = 64
  var a = []
  for(var i = 0; i < l; i++){
    if(restricted === false){
      a.push(Math.floor(Math.random()*255))
    } else {
      a.push(Math.floor(Math.random()*25)+65)
    }
  }
  var cipher = crypto.createCipher('aes256', '12345')
  var encrypted = cipher.update(Buffer.from(a).toString('base64'), 'base64', 'base64')
  encrypted += cipher.final('base64')
  return encrypted
}
