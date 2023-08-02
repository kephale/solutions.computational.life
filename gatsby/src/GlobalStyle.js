// src/GlobalStyle.js

import { createGlobalStyle } from 'styled-components';

const GlobalStyle = createGlobalStyle`
  body {
    background-color: ${({ theme }) => theme.colors.background};
    /* You can also add other global styles here */
  }
`;

export default GlobalStyle;
