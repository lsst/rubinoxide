build()
{
  default_build
  make install
  rm -r target
}
install()
{
  mkdir -p "$PREFIX"  
  cp -a ./ "$PREFIX"
  rm -rf "$PREFIX/UPS"
  install_ups
}
