import React from "react"
// import React from 'react';
import { ThemeProvider } from 'styled-components';
import theme from '../../theme';
import GlobalStyle from '../../GlobalStyle';

export const wrapRootElement = ({ element }) => (
  <ThemeProvider theme={theme}>
    <GlobalStyle />
    {element}
  </ThemeProvider>
);

const Home = () => {
  return (
      <p>Welcome to solutions.computational.life! Go to <a href="https://solutions.computational.life/about">for more information about this catalog</a> and <a href="https://solutions.computational.life/catalog">for the catalog listing</a>.</p>
  )
}

export default Home
