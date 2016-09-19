var fs = require('fs')
var Jimp = require('jimp')

Jimp.read('./test.png', function(err,image){
  console.log(image)
  var w = image.bitmap.width
  var h = image.bitmap.height
  console.log(w,h, image.bitmap.data.length)
  var values = []
  for(var i = 0; i < image.bitmap.data.length; i+=4){
    var x = (i/4) % w
    var y = Math.floor((i/4) / w)
    var value = image.bitmap.data.readUInt8(i)
    values.push([x,y,value])
    // console.log([x,y,value])
  }
  console.log(values.length)
  var n = 10000000
  console.log('slices')
  console.log(values.slice(n,n+10))

  fs.writeFileSync('./values.json', JSON.stringify(values))
})
