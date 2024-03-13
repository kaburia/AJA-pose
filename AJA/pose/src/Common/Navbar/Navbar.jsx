import React from 'react'
import './Navbar.css'
import logo from '../../Common/Images/aja.png'
import { ChatbubblesOutline, SearchOutline, SettingsOutline } from 'react-ionicons'
import { HomeOutline } from 'react-ionicons'
import { FolderOpenOutline } from 'react-ionicons'
import { PieChartOutline } from 'react-ionicons'
import { PeopleOutline } from 'react-ionicons'


const Navbar = () => {
    return (
        <>
            <nav id="navbar">
                <ul className="navbar-items flexbox-col">
                    <li className="navbar-logo flexbox-left">
                        <a className="navbar-item-inner flexbox">
                            {/* <svg
                                xmlns="http://www.w3.org/2000/svg"
                                id="Layer_1"
                                data-name="Layer 1"
                                viewBox="0 0 1438.88 1819.54"
                            >
                                <polygon points="925.79 318.48 830.56 0 183.51 1384.12 510.41 1178.46 925.79 318.48" />
                                <polygon points="1438.88 1663.28 1126.35 948.08 111.98 1586.26 0 1819.54 1020.91 1250.57 1123.78 1471.02 783.64 1663.28 1438.88 1663.28" />
                            </svg> */}
                            <img className="logo" src={logo}></img>
                        </a>
                    </li>
                    <li className="navbar-item flexbox-left">
                        <a className="navbar-item-inner flexbox-left">
                            <div className="navbar-item-inner-icon-wrapper flexbox">
                                <SearchOutline
                                    style={{color:'#fff'}}
                                    height="20px"
                                    width="20px"
                                />

                            </div>
                            <span className="link-text">Search</span>
                        </a>
                    </li>
                    <li className="navbar-item flexbox-left">
                        <a className="navbar-item-inner flexbox-left">
                            <div className="navbar-item-inner-icon-wrapper flexbox">
                            <HomeOutline
                                    style={{color:'#fff'}}
                                    height="20px"
                                    width="20px"
                                />                            </div>
                            <span className="link-text">Home</span>
                        </a>
                    </li>
                    <li className="navbar-item flexbox-left">
                        <a className="navbar-item-inner flexbox-left">
                            <div className="navbar-item-inner-icon-wrapper flexbox">
                                <FolderOpenOutline
                                    style={{color:'#fff'}}
                                    height="20px"
                                    width="20px"
                                />
                            </div>
                            <span className="link-text">Projects</span>
                        </a>
                    </li>
                    <li className="navbar-item flexbox-left">
                        <a className="navbar-item-inner flexbox-left">
                            <div className="navbar-item-inner-icon-wrapper flexbox">
                            <PieChartOutline
                                    style={{color:'#fff'}}
                                    height="20px"
                                    width="20px"
                                />                            </div>
                            <span className="link-text">Dashboard</span>
                        </a>
                    </li>
                    <li className="navbar-item flexbox-left">
                        <a className="navbar-item-inner flexbox-left">
                            <div className="navbar-item-inner-icon-wrapper flexbox">
                            <PeopleOutline
                                    style={{color:'#fff'}}
                                    height="20px"
                                    width="20px"
                                />
                            </div>
                            <span className="link-text">Team</span>
                        </a>
                    </li>
                    <li className="navbar-item flexbox-left">
                        <a className="navbar-item-inner flexbox-left">
                            <div className="navbar-item-inner-icon-wrapper flexbox">
                                <ChatbubblesOutline
                                    style={{color:'#fff'}}
                                    height="20px"
                                    width="20px"
                                />
                            </div>
                            <span className="link-text">Support</span>
                        </a>
                    </li>
                    <li className="navbar-item flexbox-left">
                        <a className="navbar-item-inner flexbox-left">
                            <div className="navbar-item-inner-icon-wrapper flexbox">
                                <SettingsOutline
                                    style={{color:'#fff'}}
                                    height="20px"
                                    width="20px"
                                />
                            </div>
                            <span className="link-text">Settings</span>
                        </a>
                    </li>
                </ul>
            </nav>
            {/* Main */}
            <main id="main" className="flexbox-col">
                <h2>Lorem ipsum!</h2>
                <p>
                    Lorem ipsum dolor sit amet, consectetur adipisicing elit. Eum corporis,
                    rerum doloremque iste sed voluptates omnis molestias molestiae animi
                    recusandae labore sit amet delectus ad necessitatibus laudantium qui!
                    Magni quisquam illum quaerat necessitatibus sint quibusdam perferendis!
                    Aut ipsam cumque deleniti error perspiciatis iusto accusamus consequuntur
                    assumenda. Obcaecati minima sed natus?
                </p>
            </main>
        </>

    )
}

export default Navbar